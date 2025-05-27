import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# Cached Dataset Loader
@st.cache_data
def load_dataset(file):
    return pd.read_csv(file)

# Cached Dummy Location Generator
@st.cache_data
def generate_dummy_locations(true_lat, true_lon, k=5, radius=0.01):
    locations = [(true_lat, true_lon)]  # Include true location
    for _ in range(k - 1):
        lat_offset = random.uniform(-radius, radius)
        lon_offset = random.uniform(-radius, radius)
        dummy_lat = true_lat + lat_offset
        dummy_lon = true_lon + lon_offset
        locations.append((dummy_lat, dummy_lon))
    return locations

# Distance Calculation
def distance(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Cached Trajectory Tracker
@st.cache_data
def track_trajectory(observed_locations):
    tracked_trajectory = []
    current_location = observed_locations[0][0]  # Start with the first true location
    tracked_trajectory.append(current_location)

    for timestamp_locations in observed_locations[1:]:
        next_location = min(timestamp_locations, key=lambda loc: distance(current_location, loc))
        tracked_trajectory.append(next_location)
        current_location = next_location  # Update for next step
    
    return tracked_trajectory

# Streamlit App
st.title("Dummy Location Privacy Mechanism with Adversarial Simulation")

# Tabs for Navigation
tabs = st.tabs(["Upload Dataset", "Generate Dummy Locations", "Simulate Adversarial Attack"])

# Tab 1: Upload Dataset
with tabs[0]:
    st.header("Step 1: Upload Dataset")
    uploaded_file = st.file_uploader("Upload Gowalla Dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        gowalla_checkins = load_dataset(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(gowalla_checkins.head())

# Tab 2: Generate Dummy Locations
with tabs[1]:
    st.header("Step 2: Generate Dummy Locations")
    if uploaded_file is not None:
        k = st.slider("Number of Locations (1 Real + k-1 Dummy)", min_value=1, max_value=5000, value=5)
        radius = st.slider("Radius for Dummy Locations (in degrees)", min_value=0.001, max_value=0.9, value=0.01)

        gowalla_checkins['dummy_locations'] = gowalla_checkins.apply(
            lambda row: generate_dummy_locations(row['latitude'], row['longitude'], k, radius), axis=1
        )
        st.write("Dataset with Dummy Locations:")
        st.dataframe(gowalla_checkins[['latitude', 'longitude', 'dummy_locations']].head())

        processed_csv = gowalla_checkins.to_csv(index=False).encode('utf-8')
        st.download_button("Download Dataset with Dummy Locations", data=processed_csv, file_name="gowalla_with_dummies.csv", mime="text/csv")

# Tab 3: Simulate Adversarial Attack
with tabs[2]:
    st.header("Step 3: Simulate Adversarial Attack")
    if uploaded_file is not None and 'dummy_locations' in gowalla_checkins:
        st.write("Simulating trajectory tracking attack...")

        # True and Tracked Trajectory Simulation
        observed_locations = gowalla_checkins['dummy_locations'].iloc[:3].tolist()
        true_trajectory = [(row['latitude'], row['longitude']) for _, row in gowalla_checkins.iloc[:3].iterrows()]
        tracked_trajectory = track_trajectory(observed_locations)

        # Calculate Errors
        errors = [distance(true, tracked) for true, tracked in zip(true_trajectory, tracked_trajectory)]
        avg_error = np.mean(errors)
        st.write(f"**Average Tracking Error for Selected Dummy Locations:** {avg_error:.4f}")

        # Plot True vs Tracked Trajectories
        true_lats, true_lons = zip(*true_trajectory)
        tracked_lats, tracked_lons = zip(*tracked_trajectory)

        plt.figure(figsize=(8, 6))
        plt.plot(true_lons, true_lats, 'g-o', label='True Trajectory')
        plt.plot(tracked_lons, tracked_lats, 'r--x', label='Tracked Trajectory')
        plt.title("True vs. Tracked Trajectories")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Add Error Rate vs k Graph
        st.subheader("Expected Error Rate vs Number of Dummy Locations (k-1)")
        k_values = list(range(2, 5000))  # Range of k values (2 to 10)

        # Calculate Expected Error Rates for Different k
        error_rates = []
        for k in k_values:
            gowalla_checkins['dummy_locations'] = gowalla_checkins.apply(
                lambda row: generate_dummy_locations(row['latitude'], row['longitude'], k, radius), axis=1
            )
            observed_locations = gowalla_checkins['dummy_locations'].iloc[:3].tolist()
            true_trajectory = [(row['latitude'], row['longitude']) for _, row in gowalla_checkins.iloc[:3].iterrows()]
            tracked_trajectory = track_trajectory(observed_locations)

            # Calculate average error for this k
            errors = [distance(true, tracked) for true, tracked in zip(true_trajectory, tracked_trajectory)]
            avg_error = np.mean(errors)
            error_rates.append(avg_error)

        # Plot Error Rate vs Number of Dummy Locations
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, error_rates, 'b-o', label='Error Rate')
        plt.title("Expected Error Rate vs Number of Dummy Locations")
        plt.xlabel("Number of Dummy Locations (k-1)")
        plt.ylabel("Average Tracking Error")
        plt.grid()
        plt.legend()
        st.pyplot(plt)

        # Add Error Rate vs Radius Graph
        st.subheader("Expected Error Rate vs Radius for Dummy Locations")

        # Generate up to 12 different radius values using linspace
        num_points = st.slider("Number of Radius Points", min_value=1, max_value=5000, value=5000)
        radius_values = np.linspace(0.001, 0.1, num_points)

        # Calculate Expected Error Rates for Different radius values
        radius_error_rates = []
        for radius in radius_values:
            gowalla_checkins['dummy_locations'] = gowalla_checkins.apply(
                lambda row: generate_dummy_locations(row['latitude'], row['longitude'], k, radius), axis=1
            )
            observed_locations = gowalla_checkins['dummy_locations'].iloc[:3].tolist()
            true_trajectory = [(row['latitude'], row['longitude']) for _, row in gowalla_checkins.iloc[:3].iterrows()]
            tracked_trajectory = track_trajectory(observed_locations)

            # Calculate average error for this radius
            errors = [distance(true, tracked) for true, tracked in zip(true_trajectory, tracked_trajectory)]
            avg_error = np.mean(errors)
            radius_error_rates.append(avg_error)

        # Plot Error Rate vs Radius
        plt.figure(figsize=(8, 6))
        plt.plot(radius_values, radius_error_rates, 'r-o', label='Error Rate')
        plt.title("Expected Error Rate vs Radius for Dummy Locations")
        plt.xlabel("Radius (degrees)")
        plt.ylabel("Average Tracking Error")
        plt.grid()
        plt.legend()
        st.pyplot(plt)
