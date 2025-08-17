import streamlit as st
import os
import json
import googlemaps

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- BÖLÜM 1: API ANAHTARLARI VE İSTEMCİ KURULUMU ---
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not openai_api_key or not google_api_key:
    st.error("Please set your OpenAI and Google Maps API keys in .streamlit/secrets.toml")
    st.stop()

gmaps = googlemaps.Client(key=google_api_key)

# --- BÖLÜM 2: YARDIMCI FONKSİYONLAR ---
@st.cache_data
def get_city_center(_gmaps: googlemaps.Client, location_name: str):
    """Finds the geographic center coordinates of a given city name."""
    print(f"--- Calling the Geocode API: {location_name} ---")
    try:
        geocode_result = _gmaps.geocode(location_name)
        if not geocode_result: return None, f"'{location_name}' could not be found on the map."
        return geocode_result[0]['geometry']['location'], None
    except Exception as e: return None, f"Geocode process failed: {str(e)}"

def format_price_level(price_level):
    """It is used to display the price levels of restaurants in Google as $$ in the code."""
    if price_level is None:
        return ""
    if price_level == 0:
        return "Free"
    return "$" * int(price_level)

def create_location_entry(place):
    """ Takes a single 'place' object from the Google Places API result and formats it
    into a standardized, clean dictionary for our application."""
    place_id = place.get('place_id')
    link = f"https://www.google.com/maps/search/?api=1&query=Google&query_place_id={place_id}" if place_id else "#"
    price_level = place.get('price_level')
    
    return {
        "name": place.get('name', 'Unknown'),
        "rating": place.get('rating'),
        "price_label": format_price_level(price_level),
        "coords": place.get('geometry', {}).get('location'),
        "link": link
    }

# --- BÖLÜM 3: ARAÇLARIN TANIMLANMASI ---

# --- SECTION 3: TOOL DEFINITIONS ---

@tool
def find_traditional_dishes_deep(city: str) -> str:
    """
    WEB SEARCH TOOL: Finds a comprehensive list of famous local dishes for any city.
    This tool acts as the first step in the planning chain to gather culinary intelligence.
    """
    print(f"--- Performing deep culinary search for: {city} ---")
    try:
        # Use DuckDuckGo to search for the most famous local dishes.
        search = DuckDuckGoSearchRun()
        query = f"most famous 15 traditional local dishes, food, and desserts in {city}"
        results = search.run(query)
        
        # Use a small LLM to parse the search results and extract a clean, comma-separated list.
        llm_for_tool = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
        processing_prompt = f"From the following text, extract at least 10-15 famous local dish names for {city}. Return ONLY a comma-separated list: {results}"
        response = llm_for_tool.invoke(processing_prompt)
        return response.content.strip()
    except Exception as e:
        return f"An error occurred while searching the web for dishes: {str(e)}"

@tool
def create_enriched_discovery_plan(city: str, dish_names: str) -> str:
    """
    STANDARD PLAN TOOL: Creates a general discovery plan with a map.
    This is the default planning tool used when no specific budget or interest is mentioned by the user.
    """
    print(f"--- Creating Enriched Plan with Links for {city} ---")
    # First, get the geographical center of the city to ensure all searches are relevant.
    city_center_coords, error = get_city_center(gmaps, city)
    if error: return json.dumps({"error": error})
    
    try:
        city_radius = 15000 # Set a wide search radius of 15km.
        
        # Find the most popular tourist attractions near the city center.
        attraction_results = gmaps.places_nearby(location=city_center_coords, radius=city_radius, type='tourist_attraction')
        attraction_suggestions = sorted([p for p in attraction_results.get('results', []) if p.get('user_ratings_total', 0) > 500], key=lambda x: x.get('user_ratings_total', 0), reverse=True)[:10]

        # Find restaurants that serve the local dishes found by the web search tool.
        dish_list = [dish.strip() for dish in dish_names.split(',')]
        all_restaurants = {}
        for dish in dish_list:
            if not dish: continue
            nearby_results = gmaps.places_nearby(location=city_center_coords, radius=city_radius, keyword=dish)
            for place in nearby_results.get('results', []):
                if place.get('place_id'): all_restaurants[place['place_id']] = place
        restaurant_suggestions = sorted([p for p in all_restaurants.values() if p.get('rating') and p.get('user_ratings_total', 0) > 100], key=lambda x: x['rating'], reverse=True)[:10]

        if not attraction_suggestions and not restaurant_suggestions:
            return json.dumps({"error": f"Could not find popular attractions or culinary spots in {city}."})

        # Select the most popular attraction as the "anchor point" for the map view.
        anchor_point_place = attraction_suggestions[0] if attraction_suggestions else restaurant_suggestions[0]
        
        # Assemble all found locations into a single, structured JSON output.
        unified_plan = {
            "location": city,
            "city_center": {"name": f"{city} City Center", "coords": city_center_coords},
            "anchor_point": create_location_entry(anchor_point_place),
            "restaurant_suggestions": [create_location_entry(p) for p in restaurant_suggestions],
            "attraction_suggestions": [create_location_entry(p) for p in attraction_suggestions]
        }
        
        return json.dumps({"discovery_plan": unified_plan}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"An error occurred while creating the plan: {str(e)}"})
    
@tool
def create_budget_focused_plan(city: str, dish_names: str, budget: str) -> str:
    """
    BUDGET PLAN TOOL: Creates a discovery plan tailored to a specific budget (e.g., 'cheap', 'luxury').
    This tool is selected when the user mentions keywords like 'budget', 'cheap', or 'expensive'.
    """
    print(f"--- Creating BUDGET-FOCUSED Plan for {city} (Budget: {budget}) ---")
    city_center_coords, error = get_city_center(gmaps, city) # Corrected function call
    if error: return json.dumps({"error": error})
    
    # Translate user's budget term into Google's price level parameters (0-4).
    price_params = {}
    budget_normalized = budget.lower()
    if "cheap" in budget_normalized or "affordable" in budget_normalized:
        price_params['max_price'] = 2 # Levels 0, 1, 2
    elif "luxury" in budget_normalized or "expensive" in budget_normalized:
        price_params['min_price'] = 3 # Levels 3, 4

    try:
        city_radius = 15000
        # Find tourist attractions (this search is not affected by budget).
        attraction_results = gmaps.places_nearby(location=city_center_coords, radius=city_radius, type='tourist_attraction')
        attraction_suggestions = sorted([p for p in attraction_results.get('results', []) if p.get('user_ratings_total', 0) > 500], key=lambda x: x.get('user_ratings_total', 0), reverse=True)[:5]

        # Search for restaurants, now including the price level filters.
        dish_list = [dish.strip() for dish in dish_names.split(',')]
        all_restaurants = {}
        for dish in dish_list:
            if not dish: continue
            search_params = {
                'location': city_center_coords, 'radius': city_radius,
                'keyword': dish, 'type': 'restaurant', **price_params
            }
            nearby_results = gmaps.places_nearby(**search_params)
            for place in nearby_results.get('results', []):
                if place.get('place_id'): all_restaurants[place['place_id']] = place
        
        restaurant_suggestions = sorted([p for p in all_restaurants.values() if p.get('rating') and p.get('user_ratings_total', 0) > 50], key=lambda x: x['rating'], reverse=True)[:7]

        if not restaurant_suggestions:
            return json.dumps({"error": f"Could not find restaurants in {city} for the specified budget '{budget}'."})

        # Assemble the final JSON output, which will include price labels.
        anchor_point_place = attraction_suggestions[0] if attraction_suggestions else restaurant_suggestions[0]
        unified_plan = {
            "location": city,
            "city_center": {"name": f"{city} City Center", "coords": city_center_coords},
            "anchor_point": create_location_entry(anchor_point_place),
            "restaurant_suggestions": [create_location_entry(p) for p in restaurant_suggestions],
            "attraction_suggestions": [create_location_entry(p) for p in attraction_suggestions]
        }
        return json.dumps({"discovery_plan": unified_plan}, indent=2)
        
    except Exception as e:
        print(f"ERROR in create_budget_focused_plan: {e}")
        return json.dumps({"error": f"An error occurred while creating the budget plan: {str(e)}"})

@tool
def create_interest_focused_plan(city: str, dish_names: str, interest: str) -> str:
    """
    INTEREST PLAN TOOL: Creates a discovery plan tailored to a user's specific interest (e.g., 'art', 'history', 'nature').
    It uses a text search with a location bias to find the most relevant places.
    """
    print(f"--- Creating ROBUST-SEARCH INTEREST-FOCUSED Plan for {city} (Interest: {interest}) ---")
    city_center_coords, error = get_city_center(gmaps, city)
    if error: 
        return json.dumps({"error": error})

    try:
        # Create a text query and a geographical bias for the search.
        query = f"{interest} in {city}"
        city_radius = 25000

        print(f"--- Searching for attractions with query='{query}' biased towards {city_center_coords} ---")
        
        # Use Google's text search ('places') with a location and radius to prioritize local results.
        places_result = gmaps.places(
            query=query, 
            location=city_center_coords,
            radius=city_radius
        )
        
        # Filter out irrelevant results like hotels or apartments.
        filtered_places = []
        for place in places_result.get('results', []):
            place_name_lower = place.get('name', '').lower()
            if 'hotel' not in place_name_lower and 'otel' not in place_name_lower and 'apart' not in place_name_lower:
                filtered_places.append(place)
        
        attraction_suggestions = sorted(
            [p for p in filtered_places if p.get('user_ratings_total', 0) > 10],
            key=lambda x: x.get('user_ratings_total', 0), 
            reverse=True
        )[:10]

        if not attraction_suggestions:
            return f"Error: Could not find any relevant attractions in {city} for the specified interest '{interest}'."

        # Find restaurants near the most popular interest-based attraction.
        dish_list = [dish.strip() for dish in dish_names.split(',')]
        all_restaurants = {}
        anchor_point_coords = attraction_suggestions[0]['geometry']['location']
        for dish in dish_list:
            if not dish: continue
            nearby_results = gmaps.places_nearby(location=anchor_point_coords, radius=2000, keyword=dish, type='restaurant')
            for place in nearby_results.get('results', []):
                if place.get('place_id'): all_restaurants[place['place_id']] = place
        restaurant_suggestions = sorted([p for p in all_restaurants.values() if p.get('rating') and p.get('user_ratings_total', 0) > 500], key=lambda x: x['rating'], reverse=True)[:5]

        # Assemble the final JSON output.
        anchor_point_place = attraction_suggestions[0]
        unified_plan = {
            "location": city,
            "city_center": {"name": f"{city} City Center", "coords": city_center_coords},
            "anchor_point": create_location_entry(anchor_point_place),
            "restaurant_suggestions": [create_location_entry(p) for p in restaurant_suggestions],
            "attraction_suggestions": [create_location_entry(p) for p in attraction_suggestions]
        }
        
        return json.dumps({"discovery_plan": unified_plan}, indent=2)
        
    except Exception as e:
        print(f"ERROR in create_interest_focused_plan: {e}")
        return f"Error: An error occurred while creating the interest-based plan: {str(e)}"

# --- BÖLÜM 4: AGENT OLUŞTURMA ---
@st.cache_resource
def get_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=openai_api_key)
    tools = [
        find_traditional_dishes_deep, 
        create_interest_focused_plan,
        create_budget_focused_plan,
        create_enriched_discovery_plan
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert global travel planner. Your goal is to choose the best tool and then format the output correctly based on which tool was used.

        **TOOL USAGE STRATEGY (in order of priority):**
        1.  If the user mentions an INTEREST, use `create_interest_focused_plan` after finding dishes.
        2.  If the user mentions a BUDGET, use `create_budget_focused_plan` after finding dishes.
        3.  If the user asks for a general plan, use `create_enriched_discovery_plan` after finding dishes.
        4.  **THE FALLBACK (B-PLAN) RULE:** If a specific tool (interest or budget) returns an "Error", acknowledge it and then call the general `create_enriched_discovery_plan` tool as a fallback.

        **RESPONSE GENERATION (CRITICAL - READ CAREFULLY):**
        - After the final successful tool call, craft a friendly and detailed text summary.
        - For each location, you MUST create a clickable Markdown link.
        
        **CONDITIONAL FORMATTING RULES:**
        -   **IF you used the `create_budget_focused_plan` tool:** For each RESTAURANT, you MUST display its price level. Use this format: `*   **[Name](Link)** (Rating: [Rating] ⭐) - Price: **[Price]**`
        -   **IF you used `create_enriched_discovery_plan` OR `create_interest_focused_plan`:** You **MUST NOT** display the price level. Use this format: `*   **[Name](Link)** (Rating: [Rating] ⭐)`

        - Your `Final Answer` should ONLY be this generated Markdown text.
        """),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# --- BÖLÜM 5: STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Global Discovery Assistant", layout="wide", page_icon="🗺️")
st.title("Global Discovery Assistant 🗺️")
st.markdown("Ask for a discovery plan for any city, and I'll create a custom plan and map for you!")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    with open('map_template.html', 'r', encoding='utf-8') as f:
        MAP_TEMPLATE = f.read()
except FileNotFoundError:
    st.error("`map_template.html` not found. Please ensure it is in the same directory as `app.py`.")
    st.stop()

# Önceki sohbet mesajlarını ve haritalarını göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if "map_data_str" in message and message["map_data_str"]:
            with st.expander("🗺️ View Unified Discovery Map"):
                html_with_data = MAP_TEMPLATE.replace('%JSON_DATA%', message["map_data_str"])
                st.components.v1.html(html_with_data, height=600, scrolling=False)

# Kullanıcıdan yeni girdi al
if query := st.chat_input("Which city would you like to explore? (e.g., 'a cheap plan for Rome')"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Researching the city's secrets and creating your custom discovery plan..."):
            agent_executor = get_agent()
            response = agent_executor.invoke({"input": query})
            
            text_response = response.get('output', "I'm sorry, I couldn't generate a response.")
            map_data_str = None
            if 'intermediate_steps' in response and response['intermediate_steps']:
                action, tool_output = response['intermediate_steps'][-1]
                
                # Harita verisi üreten tüm araçları kontrol et
                if action.tool in ['create_enriched_discovery_plan', 'create_budget_focused_plan', 'create_interest_focused_plan'] and isinstance(tool_output, str) and tool_output.strip().startswith('{'):
                    map_data_str = tool_output
            
            st.markdown(text_response, unsafe_allow_html=True)

            if map_data_str:
                with st.expander("🗺️ View Unified Discovery Map", expanded=True):
                    html_with_data = MAP_TEMPLATE.replace('%JSON_DATA%', map_data_str)
                    st.components.v1.html(html_with_data, height=600, scrolling=False)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": text_response,
                "map_data_str": map_data_str
            })