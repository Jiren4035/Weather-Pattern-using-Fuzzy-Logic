import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Create fuzzy system
def create_fuzzy_system():
    # Define fuzzy input: Rain Probability (0–100%)
    rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')

    # Define fuzzy input: Soil Moisture (0.0–1.0)
    soil = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'soil')

    # Define fuzzy output: Water Amount (liters/hour)
    water = ctrl.Consequent(np.arange(0, 11, 0.1), 'water')

    # Membership functions for Rain
    rain['low'] = fuzz.trimf(rain.universe, [0, 0, 30])
    rain['medium'] = fuzz.trimf(rain.universe, [20, 50, 80])
    rain['high'] = fuzz.trimf(rain.universe, [60, 100, 100])

    # Membership functions for Soil Moisture
    soil['dry'] = fuzz.trimf(soil.universe, [0.0, 0.0, 0.3])
    soil['normal'] = fuzz.trimf(soil.universe, [0.2, 0.5, 0.8])
    soil['wet'] = fuzz.trimf(soil.universe, [0.6, 1.0, 1.0])

    # Membership functions for Water Output
    water['none'] = fuzz.trimf(water.universe, [0, 0, 1])
    water['low'] = fuzz.trimf(water.universe, [1, 3, 5])
    water['medium'] = fuzz.trimf(water.universe, [4, 6, 8])
    water['high'] = fuzz.trimf(water.universe, [7, 10, 10])

    # Define fuzzy rules
    rule1 = ctrl.Rule(rain['high'], water['none'])                       # No watering if high chance of rain
    rule2 = ctrl.Rule(soil['wet'], water['none'])                        # No watering if soil is already wet
    rule3 = ctrl.Rule(rain['low'] & soil['dry'], water['high'])         # Water heavily if rain is low and soil dry
    rule4 = ctrl.Rule(rain['low'] & soil['normal'], water['medium'])    # Medium watering if rain is low, soil normal
    rule5 = ctrl.Rule(rain['medium'] & soil['dry'], water['medium'])    # Medium watering if rain is medium, soil dry
    rule6 = ctrl.Rule(rain['medium'] & soil['normal'], water['low'])    # Low watering if rain is medium, soil normal

    # Create control system and simulation
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    simulation = ctrl.ControlSystemSimulation(system)

    # Return fuzzy system and variable references
    return simulation, rain, soil, water

# Evaluate the fuzzy system with specific inputs
def evaluate_irrigation(simulation, rain_input, soil_input):
    # Check input ranges
    if not (0 <= rain_input <= 100) or not (0.0 <= soil_input <= 1.0):
        raise ValueError("Rain must be between 0 and 100, and Soil Moisture must be between 0.0 and 1.0.")

    # Assign inputs to the fuzzy system
    simulation.input['rain'] = rain_input
    simulation.input['soil'] = soil_input

    # Compute the fuzzy result
    simulation.compute()

    # Return the crisp output for water
    return simulation.output['water']

# Plot membership functions for rain, soil, and water
def plot_membership_functions(rain, soil, water):
    fig, axs = plt.subplots(nrows=3, figsize=(10, 10))
    rain.view(ax=axs[0])
    soil.view(ax=axs[1])
    water.view(ax=axs[2])
    plt.tight_layout()
    plt.show()

# Main function to run the program
def main():
    # Create fuzzy irrigation system
    sim, rain_var, soil_var, water_var = create_fuzzy_system()

    # Prompt user for input
    try:
        # Ask for rain input
        print("Enter Rain Probability (0–100%):")
        rain_input = float(input("Rain %: "))

        # Ask for soil moisture input
        print("Enter Soil Moisture Level (0.0–1.0):")
        soil_input = float(input("Soil Moisture: "))

        # Run evaluation
        result = evaluate_irrigation(sim, rain_input, soil_input)
        print(f"\nRain = {rain_input}%, Soil = {soil_input}")
        print(f"→ Recommended Watering: {result:.2f} liters/hour\n")

        # Display defuzzified result
        water_var.view(sim=sim)
        plt.title("Output: Irrigation Recommendation")
        plt.show()

        # Plot all membership functions
        plot_membership_functions(rain_var, soil_var, water_var)

    except ValueError as e:
        # Handle invalid input
        print(f"Input Error: {e}")

# Entry point
if __name__ == "__main__":
    main()
