# ================================================
# fuzzy_danger_level.py
# ================================================
import numpy as np
import skfuzzy as fuzz

# Define danger scale: 0 to 100
x_danger = np.arange(0, 101, 1)

# Define fuzzy membership functions
low = fuzz.trimf(x_danger, [0, 0, 40])       # Safe or friendly
medium = fuzz.trimf(x_danger, [30, 50, 70])  # Can harm moderately
high = fuzz.trapmf(x_danger, [60, 75, 100, 100])  # Dangerous or predator

# Mapping of animals to danger fuzzy category (based on behavior)
animal_categories = {
    'Zebra': 'medium',
    'Tiger': 'high',
    'Rhinoceros': 'high',
    'Ostrich': 'medium',
    'Lion': 'high',
    'Leopard': 'high',
    'Horse': 'medium',
    'Jaguar': 'high',
    'Harbor seal': 'medium',
    'Goat': 'low',
    'Giraffe': 'medium',
    'Fox': 'medium',
    'Elephant': 'high',
    'Eagle': 'medium',
    'Deer': 'low',
    'Crab': 'low',
    'Chicken': 'low',
    'Caterpillar': 'low',
    'Cheetah': 'high',
    'Butterfly': 'low'
}

def compute_danger_level(animal_name: str) -> float:
    """
    Given an animal name, compute the crisp danger level (%) using fuzzy logic.
    """

    animal_name = animal_name.strip().title()

    if animal_name not in animal_categories:
        return 0.0

    category = animal_categories[animal_name]

    # Generate fuzzy values
    if category == 'low':
        danger_fuzzy = fuzz.interp_membership(x_danger, low, 25)
        crisp_value = 25
    elif category == 'medium':
        danger_fuzzy = fuzz.interp_membership(x_danger, medium, 55)
        crisp_value = 55
    elif category == 'high':
        danger_fuzzy = fuzz.interp_membership(x_danger, high, 85)
        crisp_value = 85
    else:
        danger_fuzzy = 0
        crisp_value = 0

    return round(crisp_value, 2)

# Example usage (for testing)
if __name__ == "__main__":
    animals = [
        'Zebra', 'Tiger', 'Rhinoceros', 'Ostrich', 'Lion', 'Leopard', 'Horse',
        'Jaguar', 'Harbor seal', 'Goat', 'Giraffe', 'Fox', 'Elephant',
        'Eagle', 'Deer', 'Crab', 'Chicken', 'Caterpillar', 'Cheetah', 'Butterfly'
    ]

    print("=== Fuzzy Danger Level of Animals ===")
    for a in animals:
        print(f"{a:15s} ➤ {compute_danger_level(a)} %")
