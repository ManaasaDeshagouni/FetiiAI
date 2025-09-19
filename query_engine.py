# query_engine.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

# Load data from preprocessed Fetii dataset
def load_and_prepare_data(excel_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    
    trip_df = xls.parse("Trip Data")
    rider_df = xls.parse("Checked in User ID's")
    demo_df = xls.parse("Customer Demographics")
    
    # Clean columns
    trip_df.rename(columns={"Trip ID": "trip_id", "Booking User ID": "booker_id", 
                            "Trip Date and Time": "trip_time", "Total Passengers": "total_passengers"}, inplace=True)
    rider_df.rename(columns={"Trip ID": "trip_id", "User ID": "user_id"}, inplace=True)
    demo_df.rename(columns={"User ID": "user_id", "Age": "age"}, inplace=True)
    
    # Merge
    riders_with_age = pd.merge(rider_df, demo_df, on="user_id", how="left")
    riders_full = pd.merge(riders_with_age, trip_df, on="trip_id", how="left")
    
    # Add time features
    riders_full['trip_time'] = pd.to_datetime(riders_full['trip_time'])
    riders_full['hour'] = riders_full['trip_time'].dt.hour
    riders_full['day_of_week'] = riders_full['trip_time'].dt.day_name()
    riders_full['is_weekend'] = riders_full['day_of_week'].isin(["Saturday", "Sunday"])

    # Age buckets
    def age_bucket(age):
        if pd.isna(age): return "Unknown"
        age = int(age)
        if age < 18: return "<18"
        elif age <= 24: return "18–24"
        elif age <= 34: return "25–34"
        elif age <= 44: return "35–44"
        elif age <= 54: return "45–54"
        else: return "55+"
    
    riders_full['age_group'] = riders_full['age'].apply(age_bucket)

    # Clean dropoff
    riders_full['dropoff_simple'] = riders_full['Drop Off Address'].astype(str).apply(lambda x: x.split(",")[0].strip())
    
    return riders_full


# Function 1: Top drop-offs by age group and day
def top_dropoffs_by_age_group_and_day(df: pd.DataFrame, age_group: str, day: Optional[str] = None, top_k: int = 5) -> Tuple[pd.DataFrame, plt.Figure]:
    filtered = df[df["age_group"] == age_group]
    if day:
        filtered = filtered[filtered["day_of_week"] == day]
    result = (
        filtered.groupby("dropoff_simple")
        .size()
        .sort_values(ascending=False)
        .head(top_k)
        .reset_index(name="trip_count")
    )

    fig, ax = plt.subplots()
    sns.barplot(x="trip_count", y="dropoff_simple", data=result, ax=ax)
    ax.set_title(f"Top {top_k} Drop-Offs for Age Group {age_group}" + (f" on {day}" if day else ""))
    ax.set_xlabel("Trip Count")
    ax.set_ylabel("Drop-Off Location")
    plt.tight_layout()
    return result, fig


# Function 2: Peak hours for large groups
def peak_hours_for_large_groups(df: pd.DataFrame, min_group_size: int = 6) -> Tuple[pd.DataFrame, plt.Figure]:
    filtered = df[df["total_passengers"] >= min_group_size]
    hourly_counts = (
        filtered.groupby("hour")
        .size()
        .reset_index(name="trip_count")
        .sort_values("hour")
    )

    fig, ax = plt.subplots()
    sns.lineplot(data=hourly_counts, x="hour", y="trip_count", marker="o", ax=ax)
    ax.set_title(f"Peak Hours for Groups with ≥ {min_group_size} Riders")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Trip Count")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    return hourly_counts, fig


# Function 3: Number of trips to a specific location over time
def trips_to_specific_location(df: pd.DataFrame, location_name: str) -> Tuple[pd.DataFrame, plt.Figure]:
    filtered = df[df['dropoff_simple'].str.lower().str.contains(location_name.lower())]
    filtered = filtered.copy()
    filtered['trip_date'] = filtered['trip_time'].dt.date

    trend = (
        filtered.groupby("trip_date")
        .size()
        .reset_index(name="trip_count")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=trend, x="trip_date", y="trip_count", marker="o", ax=ax)
    ax.set_title(f"Trips to '{location_name.title()}' Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trip Count")
    ax.tick_params(axis='x', rotation=45)
    plt.grid(True)
    plt.tight_layout()
    return trend, fig


# Function 4: Age distribution at a specific location
def age_distribution_at_location(df: pd.DataFrame, location_name: str) -> Tuple[pd.DataFrame, plt.Figure]:
    filtered = df[df['dropoff_simple'].str.lower().str.contains(location_name.lower())]
    age_dist = (
        filtered['age_group']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'age_group', 'age_group': 'count'})
        .sort_values(by='count', ascending=False)
    )

    fig, ax = plt.subplots()
    sns.barplot(data=age_dist, x="count", y="age_group", ax=ax)
    ax.set_title(f"Age Distribution at '{location_name.title()}'")
    ax.set_xlabel("Trip Count")
    ax.set_ylabel("Age Group")
    plt.tight_layout()
    return age_dist, fig


# Function 5: Group size distribution by day of the week
def group_size_by_day_of_week(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    group_stats = (
        df.groupby("day_of_week")["total_passengers"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )

    fig, ax = plt.subplots()
    sns.barplot(data=group_stats, x="mean", y="day_of_week", ax=ax)
    ax.set_title("Average Group Size by Day of the Week")
    ax.set_xlabel("Average Group Size")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    return group_stats, fig


# Function 6: Least busy locations by day (for crowd avoidance)
def least_busy_locations_by_day(df: pd.DataFrame, day: str, min_trips: int = 5) -> Tuple[pd.DataFrame, plt.Figure]:
    """Find locations with the fewest trips on a specific day"""
    filtered = df[df["day_of_week"] == day]
    location_counts = (
        filtered.groupby("dropoff_simple")
        .size()
        .reset_index(name="trip_count")
        .sort_values("trip_count")
    )
    
    # Filter out locations with too few trips
    location_counts = location_counts[location_counts["trip_count"] >= min_trips]
    
    # Get the least busy locations (bottom 10)
    result = location_counts.head(10)
    
    fig, ax = plt.subplots()
    sns.barplot(x="trip_count", y="dropoff_simple", data=result, ax=ax)
    ax.set_title(f"Least Busy Locations on {day} (≥{min_trips} trips)")
    ax.set_xlabel("Trip Count")
    ax.set_ylabel("Location")
    plt.tight_layout()
    return result, fig


# Function 7: Busiest hours by location
def busiest_hours_by_location(df: pd.DataFrame, location_name: str) -> Tuple[pd.DataFrame, plt.Figure]:
    """Find the busiest hours for a specific location"""
    filtered = df[df['dropoff_simple'].str.lower().str.contains(location_name.lower())]
    hourly_counts = (
        filtered.groupby("hour")
        .size()
        .reset_index(name="trip_count")
        .sort_values("trip_count", ascending=False)
    )
    
    fig, ax = plt.subplots()
    sns.barplot(x="hour", y="trip_count", data=hourly_counts.head(12), ax=ax)
    ax.set_title(f"Busiest Hours at Locations Containing '{location_name.title()}'")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Trip Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return hourly_counts, fig


# Function 8: Weekend vs weekday patterns
def weekend_vs_weekday_patterns(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """Compare patterns between weekends and weekdays"""
    patterns = (
        df.groupby("is_weekend")
        .agg({
            "trip_id": "count",
            "total_passengers": "mean",
            "hour": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        })
        .reset_index()
        .rename(columns={
            "trip_id": "total_trips",
            "total_passengers": "avg_group_size", 
            "hour": "peak_hour"
        })
    )
    
    patterns["day_type"] = patterns["is_weekend"].map({True: "Weekend", False: "Weekday"})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Total trips comparison
    sns.barplot(data=patterns, x="day_type", y="total_trips", ax=ax1)
    ax1.set_title("Total Trips: Weekend vs Weekday")
    ax1.set_ylabel("Total Trips")
    
    # Average group size comparison
    sns.barplot(data=patterns, x="day_type", y="avg_group_size", ax=ax2)
    ax2.set_title("Average Group Size: Weekend vs Weekday")
    ax2.set_ylabel("Average Group Size")
    
    plt.tight_layout()
    return patterns, fig


# Function 9: Quietest day overall (by trip count)
def quietest_day_overall(df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """Return trip counts by day of week (ascending) and a simple bar chart.
    The first row represents the quietest day by ride count.
    """
    counts = (
        df.groupby("day_of_week")
        .size()
        .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])  # natural order
        .reset_index(name="trip_count")
        .sort_values("trip_count", ascending=True)
    )

    fig, ax = plt.subplots()
    sns.barplot(data=counts, x="trip_count", y="day_of_week", ax=ax, palette="crest")
    ax.set_title("Trips by Day of Week (Quietest at top)")
    ax.set_xlabel("Trip Count")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    return counts, fig


# === LLM-Native Functions ===

import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Load environment variable
load_dotenv()
# Prefer Streamlit secrets, then fall back to environment variable
OPENAI_API_KEY = None
try:
    if hasattr(st, "secrets"):
        OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")
except Exception:
    OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def friendly_response(raw_text: str) -> str:
    """
    Rewrite raw stats/technical text in a user-friendly way.
    """
    if not client:
        return "Sorry, LLM service is not available."

    prompt = f"""
You are Assistentee, a friendly rideshare data assistant.
Your job is to rewrite this technical explanation into something friendly, casual, and easy to understand.

Raw text:
{raw_text}

Rewrite it:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You rewrite technical data explanations into friendly summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in friendly_response: {e}")
        return "Oops, I couldn't simplify that right now."


def natural_summary(df: pd.DataFrame, context: str = "") -> str:
    """
    Generate a natural-language paragraph summary of a dataframe.
    Useful for chart or table explanations.
    """
    if not client:
        return "Sorry, LLM service is not available."

    # Create a more concise summary of the data
    if len(df) == 0:
        return "No data found for this query."
    
    # Get key insights from the data
    top_items = df.head(3)
    if 'trip_count' in df.columns:
        total_trips = df['trip_count'].sum()
        top_location = df.iloc[0]['dropoff_simple'] if 'dropoff_simple' in df.columns else df.iloc[0].iloc[0]
        top_count = df.iloc[0]['trip_count']
        
        # Create a simple, natural summary without specific numbers
        if context and "dropoff" in context.lower():
            return f"Looks like most riders head to **{top_location}** and other popular spots. Here are the top destinations:"
        elif context and "peak" in context.lower():
            peak_hour = df.loc[df['trip_count'].idxmax(), 'hour']
            time_str = f"{peak_hour}:00" if peak_hour < 12 else f"{peak_hour-12}:00 PM" if peak_hour > 12 else "12:00 PM"
            return f"Large groups ride most around **{time_str}** and other peak hours. Here's the hourly breakdown:"
        elif context and "trip" in context.lower():
            return f"Found quite a few trips to this location! Here's the trend over time:"
        elif context and "age" in context.lower():
            top_age = df.iloc[0]['age_group']
            return f"**{top_age}** is the most common age group here. Here's the full breakdown:"
        elif context and "group" in context.lower():
            largest_day = df.loc[df['mean'].idxmax(), 'day_of_week']
            return f"**{largest_day}** has the largest average group sizes. Here's the breakdown by day:"
    
    # Fallback to GPT-3.5 for complex summaries (faster than GPT-4)
    try:
        sample_data = df.head(5).to_markdown()
    except ImportError:
        # Fallback if tabulate is not available
        sample_data = df.head(5).to_string()

    prompt = f"""
You are Assistentee, a friendly data assistant. Summarize this rideshare data in 1-2 sentences, like you're talking to a friend.

Context: {context}
Data: {sample_data}

Keep it casual and conversational. Don't include specific numbers in the summary - just mention the main trend or pattern. Be natural and friendly.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster than GPT-4
            messages=[
                {"role": "system", "content": "You summarize data insights like a friendly analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in natural_summary: {e}")
        return "Here's what I found in the data:"