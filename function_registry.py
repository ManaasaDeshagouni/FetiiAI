# Function Registry for Enhanced GPT Router 2.0
# This defines all available functions with their capabilities, signatures, and descriptions

from typing import Dict, List, Any, Callable
import pandas as pd
import matplotlib.pyplot as plt

# Function registry with full descriptions for LLM introspection
FUNCTION_REGISTRY = {
    "top_dropoffs_by_age_group_and_day": {
        "name": "top_dropoffs_by_age_group_and_day",
        "purpose": "Find the most popular dropoff locations for a specific age group on a specific day",
        "description": "Analyzes where riders of a certain age group go most frequently on a given day. Useful for understanding demographic preferences and popular destinations.",
        "signature": {
            "age_group": "str (required) - Age group like '18–24', '25–34', '35–44', '45–54', '55+'",
            "day": "str (optional) - Day of week like 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'",
            "top_k": "int (optional, default=5) - Number of top locations to return"
        },
        "example_usage": "top_dropoffs_by_age_group_and_day(age_group='18–24', day='Friday')",
        "example_queries": [
            "Where do young people go on Friday nights?",
            "What are the top destinations for 18-24 year olds on Saturday?",
            "Where do college kids hang out on weekends?"
        ]
    },
    
    "peak_hours_for_large_groups": {
        "name": "peak_hours_for_large_groups", 
        "purpose": "Analyze when large groups of riders are most active throughout the day",
        "description": "Shows hourly patterns of when groups of a certain size or larger tend to ride. Useful for understanding group behavior patterns and peak demand times.",
        "signature": {
            "min_group_size": "int (required) - Minimum group size to analyze (e.g., 6 for groups of 6+ people)"
        },
        "example_usage": "peak_hours_for_large_groups(min_group_size=6)",
        "example_queries": [
            "When do big groups ride most?",
            "What time do large parties usually travel?",
            "When are groups of 8+ people most active?"
        ]
    },
    
    "trips_to_specific_location": {
        "name": "trips_to_specific_location",
        "purpose": "Count and analyze trips to a specific location or venue",
        "description": "Tracks how many trips were made to locations containing a specific name. Shows trends over time for popular venues.",
        "signature": {
            "location_name": "str (required) - Name or part of location name to search for (e.g., 'Moody', 'Domain', 'Rainey')"
        },
        "example_usage": "trips_to_specific_location(location_name='Moody Center')",
        "example_queries": [
            "How many rides go to the Moody Center?",
            "What's the traffic like to Domain?",
            "How popular is Rainey Street?"
        ]
    },
    
    "age_distribution_at_location": {
        "name": "age_distribution_at_location",
        "purpose": "Show the age demographics of riders going to a specific location",
        "description": "Analyzes what age groups visit a particular venue most frequently. Useful for understanding venue demographics and target audiences.",
        "signature": {
            "location_name": "str (required) - Name or part of location name to analyze"
        },
        "example_usage": "age_distribution_at_location(location_name='Shakespeare')",
        "example_queries": [
            "What ages hang out at Shakespeare's?",
            "Who goes to the Domain?",
            "What's the age breakdown at Rainey Street?"
        ]
    },
    
    "group_size_by_day_of_week": {
        "name": "group_size_by_day_of_week",
        "purpose": "Analyze average group sizes by day of the week",
        "description": "Shows how group sizes vary throughout the week. Useful for understanding when people travel in larger vs smaller groups.",
        "signature": {},
        "example_usage": "group_size_by_day_of_week()",
        "example_queries": [
            "What's the average group size by day?",
            "When do people travel in bigger groups?",
            "How do group sizes change throughout the week?"
        ]
    },
    
    "least_busy_locations_by_day": {
        "name": "least_busy_locations_by_day",
        "purpose": "Find locations with the fewest trips on a specific day (for crowd avoidance)",
        "description": "Identifies locations that receive fewer trips on a given day, useful for finding quieter spots or avoiding crowds.",
        "signature": {
            "day": "str (required) - Day of week to analyze",
            "min_trips": "int (optional, default=5) - Minimum number of trips to consider a location"
        },
        "example_usage": "least_busy_locations_by_day(day='Saturday')",
        "example_queries": [
            "Where should I avoid on Saturday because I hate crowds?",
            "What are the quietest spots on Friday night?",
            "Where can I go to avoid busy places on weekends?"
        ]
    },
    
    "busiest_hours_by_location": {
        "name": "busiest_hours_by_location",
        "purpose": "Find the busiest hours for a specific location",
        "description": "Shows when a particular venue is most crowded throughout the day. Useful for timing visits to avoid or find peak times.",
        "signature": {
            "location_name": "str (required) - Name or part of location name to analyze"
        },
        "example_usage": "busiest_hours_by_location(location_name='Domain')",
        "example_queries": [
            "When is the Domain busiest?",
            "What time should I avoid Rainey Street?",
            "When does Moody Center get crowded?"
        ]
    },
    
    "weekend_vs_weekday_patterns": {
        "name": "weekend_vs_weekday_patterns",
        "purpose": "Compare riding patterns between weekends and weekdays",
        "description": "Analyzes differences in trip patterns, popular destinations, and group sizes between weekend and weekday usage.",
        "signature": {},
        "example_usage": "weekend_vs_weekday_patterns()",
        "example_queries": [
            "How do weekends compare to weekdays?",
            "What's different about weekend vs weekday patterns?",
            "Do people ride differently on weekends?"
        ]
    }
    ,
    "quietest_day_overall": {
        "name": "quietest_day_overall",
        "purpose": "Find the quietest day by total rides across the dataset",
        "description": "Aggregates trips by day of week and surfaces the day with the fewest rides, along with a ranked list.",
        "signature": {},
        "example_usage": "quietest_day_overall()",
        "example_queries": [
            "What is the quietest day?",
            "Which day has the least rides?"
        ]
    }
}

# Function mapping to actual implementations
FUNCTION_IMPLEMENTATIONS = {
    "top_dropoffs_by_age_group_and_day": "top_dropoffs_by_age_group_and_day",
    "peak_hours_for_large_groups": "peak_hours_for_large_groups", 
    "trips_to_specific_location": "trips_to_specific_location",
    "age_distribution_at_location": "age_distribution_at_location",
    "group_size_by_day_of_week": "group_size_by_day_of_week",
    "least_busy_locations_by_day": "least_busy_locations_by_day",
    "busiest_hours_by_location": "busiest_hours_by_location",
    "weekend_vs_weekday_patterns": "weekend_vs_weekday_patterns"
    ,
    "quietest_day_overall": "quietest_day_overall"
}

def get_function_descriptions() -> str:
    """Generate a natural language description of all available functions for the LLM"""
    descriptions = []
    for func_name, func_info in FUNCTION_REGISTRY.items():
        desc = f"""
**{func_name}**: {func_info['purpose']}
- Description: {func_info['description']}
- Parameters: {', '.join([f"{k}: {v}" for k, v in func_info['signature'].items()])}
- Example: {func_info['example_usage']}
- Good for queries like: {', '.join(func_info['example_queries'][:2])}
"""
        descriptions.append(desc)
    
    return "\n".join(descriptions)

def get_function_signatures() -> Dict[str, List[str]]:
    """Get function signatures for validation"""
    signatures = {}
    for func_name, func_info in FUNCTION_REGISTRY.items():
        signatures[func_name] = list(func_info['signature'].keys())
    return signatures
