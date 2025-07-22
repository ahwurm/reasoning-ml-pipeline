"""
Binary task templates for DDM dataset generation.

Provides a rich collection of binary decision task templates
across multiple categories and difficulty levels.
"""
import random
from typing import Dict, List, Tuple, Optional


class BinaryTaskTemplates:
    """Collection of binary decision task templates."""
    
    def __init__(self):
        """Initialize task templates."""
        self._load_fact_databases()
        self._setup_templates()
    
    def _load_fact_databases(self):
        """Load fact databases for true/false questions."""
        # Geography facts
        self.capitals = {
            "France": "Paris",
            "Germany": "Berlin",
            "Italy": "Rome",
            "Spain": "Madrid",
            "United Kingdom": "London",
            "United States": "Washington D.C.",
            "Canada": "Ottawa",
            "Japan": "Tokyo",
            "China": "Beijing",
            "India": "New Delhi",
            "Brazil": "Brasília",
            "Australia": "Canberra",
            "Russia": "Moscow",
            "Mexico": "Mexico City",
            "South Korea": "Seoul",
            "Egypt": "Cairo",
            "South Africa": "Pretoria",
            "Argentina": "Buenos Aires",
            "Sweden": "Stockholm",
            "Norway": "Oslo"
        }
        
        # Science facts
        self.elements = {
            "noble_gases": ["Helium", "Neon", "Argon", "Krypton", "Xenon", "Radon"],
            "metals": ["Iron", "Gold", "Silver", "Copper", "Aluminum", "Lead", "Zinc"],
            "nonmetals": ["Carbon", "Nitrogen", "Oxygen", "Sulfur", "Phosphorus"],
            "halogens": ["Fluorine", "Chlorine", "Bromine", "Iodine"],
            "alkali_metals": ["Lithium", "Sodium", "Potassium", "Rubidium", "Cesium"]
        }
        
        # Physics facts
        self.physics_facts = {
            "speed_of_light": 299792458,  # m/s
            "gravity_earth": 9.81,  # m/s²
            "boiling_point_water": 100,  # °C at sea level
            "freezing_point_water": 0,  # °C
            "absolute_zero_celsius": -273.15,  # °C
        }
        
        # Historical facts
        self.historical_years = {
            "World War I end": 1918,
            "World War II end": 1945,
            "American Independence": 1776,
            "French Revolution": 1789,
            "Fall of Berlin Wall": 1989,
            "Moon Landing": 1969,
            "Columbus reaches Americas": 1492,
            "Magna Carta": 1215
        }
        
        # Size comparisons
        self.sizes = {
            "planets": {
                "Jupiter": 139820,  # km diameter
                "Saturn": 116460,
                "Neptune": 49244,
                "Uranus": 50724,
                "Earth": 12742,
                "Venus": 12104,
                "Mars": 6779,
                "Mercury": 4879
            },
            "countries_area": {  # km²
                "Russia": 17098242,
                "Canada": 9984670,
                "United States": 9833517,
                "China": 9596961,
                "Brazil": 8514877,
                "Australia": 7692024,
                "India": 3287263,
                "Argentina": 2780400,
                "Kazakhstan": 2724900,
                "Algeria": 2381741
            }
        }
    
    def _setup_templates(self):
        """Set up task templates by category."""
        self.templates = {
            "math_equation": {
                "easy": [
                    {"template": "Is {a} + {b} = {c}?", "type": "addition"},
                    {"template": "Is {a} - {b} = {c}?", "type": "subtraction"},
                    {"template": "Is {a} × {b} = {c}?", "type": "multiplication"},
                ],
                "medium": [
                    {"template": "Is {a} × {b} + {c} = {d}?", "type": "mixed"},
                    {"template": "Is ({a} + {b}) × {c} = {d}?", "type": "parentheses"},
                    {"template": "Is {a}² = {b}?", "type": "square"},
                ],
                "hard": [
                    {"template": "Is {a}² + {b}² = {c}²?", "type": "pythagorean"},
                    {"template": "Is {a}³ = {b}?", "type": "cube"},
                    {"template": "Is √{a} = {b}?", "type": "sqrt"},
                ]
            },
            "comparison": {
                "easy": [
                    {"template": "Is {a} greater than {b}?", "type": "greater"},
                    {"template": "Is {a} less than {b}?", "type": "less"},
                    {"template": "Is {a} equal to {b}?", "type": "equal"},
                ],
                "medium": [
                    {"template": "Is {a} at least {b}?", "type": "gte"},
                    {"template": "Is {a} at most {b}?", "type": "lte"},
                    {"template": "Is {a} between {b} and {c}?", "type": "between"},
                ],
                "hard": [
                    {"template": "Is |{a} - {b}| < {c}?", "type": "abs_diff"},
                    {"template": "Is {a}% of {b} greater than {c}?", "type": "percentage"},
                    {"template": "Is {a}/{b} > {c}/{d}?", "type": "fraction"},
                ]
            },
            "geography": {
                "easy": [
                    {"template": "Is {city} the capital of {country}?", "type": "capital"},
                    {"template": "Is {country1} larger than {country2}?", "type": "size"},
                ],
                "medium": [
                    {"template": "Is {city} located in {country}?", "type": "location"},
                    {"template": "Does {country} border {other_country}?", "type": "border"},
                ],
                "hard": [
                    {"template": "Is {city} north of {other_city}?", "type": "direction"},
                    {"template": "Is the population of {country1} greater than {country2}?", "type": "population"},
                ]
            },
            "science": {
                "easy": [
                    {"template": "Is {element} a noble gas?", "type": "element_class"},
                    {"template": "Is water's boiling point {temp}°C at sea level?", "type": "boiling_point"},
                ],
                "medium": [
                    {"template": "Is {element} a metal?", "type": "metal_check"},
                    {"template": "Does {element} have {protons} protons?", "type": "atomic_number"},
                ],
                "hard": [
                    {"template": "Is the speed of light approximately {speed} m/s?", "type": "physics_constant"},
                    {"template": "Is {compound} an organic compound?", "type": "chemistry"},
                ]
            },
            "history": {
                "easy": [
                    {"template": "Did {event} happen in {year}?", "type": "year_check"},
                    {"template": "Did {event1} happen before {event2}?", "type": "chronology"},
                ],
                "medium": [
                    {"template": "Was {person} alive in {year}?", "type": "lifetime"},
                    {"template": "Did {event} last more than {duration} years?", "type": "duration"},
                ],
                "hard": [
                    {"template": "Were {country1} and {country2} allies in {war}?", "type": "alliance"},
                    {"template": "Did {invention} exist before {year}?", "type": "technology"},
                ]
            }
        }
    
    def generate_math_equation(self, difficulty: str) -> Dict:
        """Generate a math equation verification task."""
        template = random.choice(self.templates["math_equation"][difficulty])
        
        if difficulty == "easy":
            if template["type"] == "addition":
                a = random.randint(1, 50)
                b = random.randint(1, 50)
                correct_c = a + b
                c = correct_c if random.random() < 0.5 else correct_c + random.randint(-5, 5)
                if c == correct_c and random.random() < 0.5:
                    c += random.choice([-1, 1])
                return {
                    "prompt": template["template"].format(a=a, b=b, c=c),
                    "correct_answer": "Yes" if c == correct_c else "No",
                    "values": {"a": a, "b": b, "c": c, "correct_c": correct_c}
                }
                
        elif difficulty == "medium":
            if template["type"] == "square":
                a = random.randint(1, 20)
                correct_b = a * a
                b = correct_b if random.random() < 0.5 else correct_b + random.randint(-10, 10)
                if b == correct_b and random.random() < 0.5:
                    b += random.choice([-1, 1])
                return {
                    "prompt": template["template"].format(a=a, b=b),
                    "correct_answer": "Yes" if b == correct_b else "No",
                    "values": {"a": a, "b": b, "correct_b": correct_b}
                }
                
        # Add more template implementations as needed
        return self._default_math_task(difficulty)
    
    def generate_geography_task(self, difficulty: str) -> Dict:
        """Generate a geography fact checking task."""
        template = random.choice(self.templates["geography"][difficulty])
        
        if template["type"] == "capital":
            country = random.choice(list(self.capitals.keys()))
            correct_capital = self.capitals[country]
            
            if random.random() < 0.5:
                city = correct_capital
                is_correct = True
            else:
                # Pick a different capital
                other_capitals = [c for c in self.capitals.values() if c != correct_capital]
                city = random.choice(other_capitals)
                is_correct = False
            
            return {
                "prompt": template["template"].format(city=city, country=country),
                "correct_answer": "Yes" if is_correct else "No",
                "values": {"city": city, "country": country, "correct_capital": correct_capital}
            }
        
        elif template["type"] == "size":
            countries = list(self.sizes["countries_area"].keys())
            country1, country2 = random.sample(countries, 2)
            size1 = self.sizes["countries_area"][country1]
            size2 = self.sizes["countries_area"][country2]
            
            return {
                "prompt": template["template"].format(country1=country1, country2=country2),
                "correct_answer": "Yes" if size1 > size2 else "No",
                "values": {"country1": country1, "country2": country2, "size1": size1, "size2": size2}
            }
        
        return self._default_geography_task(difficulty)
    
    def generate_science_task(self, difficulty: str) -> Dict:
        """Generate a science fact checking task."""
        template = random.choice(self.templates["science"][difficulty])
        
        if template["type"] == "element_class":
            # Pick an element
            all_elements = []
            for category, elements in self.elements.items():
                all_elements.extend(elements)
            
            element = random.choice(all_elements)
            is_noble_gas = element in self.elements["noble_gases"]
            
            # Sometimes ask about actual noble gases, sometimes not
            if random.random() < 0.5:
                element = random.choice(self.elements["noble_gases"])
                is_noble_gas = True
            
            return {
                "prompt": template["template"].format(element=element),
                "correct_answer": "Yes" if is_noble_gas else "No",
                "values": {"element": element, "category": "noble_gas"}
            }
        
        elif template["type"] == "boiling_point":
            correct_temp = 100
            temp = correct_temp if random.random() < 0.5 else random.choice([0, 50, 80, 90, 100, 110, 200])
            
            return {
                "prompt": template["template"].format(temp=temp),
                "correct_answer": "Yes" if temp == correct_temp else "No",
                "values": {"temp": temp, "correct_temp": correct_temp}
            }
        
        return self._default_science_task(difficulty)
    
    def _default_math_task(self, difficulty: str) -> Dict:
        """Default math task generator."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        correct_sum = a + b
        given_sum = correct_sum if random.random() < 0.5 else correct_sum + random.randint(-10, 10)
        
        return {
            "prompt": f"Is {a} + {b} = {given_sum}?",
            "correct_answer": "Yes" if given_sum == correct_sum else "No",
            "values": {"a": a, "b": b, "given": given_sum, "correct": correct_sum}
        }
    
    def _default_geography_task(self, difficulty: str) -> Dict:
        """Default geography task generator."""
        country = random.choice(list(self.capitals.keys()))
        capital = self.capitals[country]
        
        return {
            "prompt": f"Is {capital} the capital of {country}?",
            "correct_answer": "Yes",
            "values": {"capital": capital, "country": country}
        }
    
    def _default_science_task(self, difficulty: str) -> Dict:
        """Default science task generator."""
        element = random.choice(self.elements["noble_gases"])
        
        return {
            "prompt": f"Is {element} a noble gas?",
            "correct_answer": "Yes",
            "values": {"element": element}
        }
    
    def get_random_task(self, category: str, difficulty: str) -> Dict:
        """Get a random task from specified category and difficulty."""
        generators = {
            "math": self.generate_math_equation,
            "geography": self.generate_geography_task,
            "science": self.generate_science_task
        }
        
        if category in generators:
            return generators[category](difficulty)
        else:
            raise ValueError(f"Unknown category: {category}")


# Extended task generators for more variety
class ExtendedBinaryTasks:
    """Extended collection of binary tasks for more variety."""
    
    @staticmethod
    def generate_even_odd_task(difficulty: str) -> Dict:
        """Generate even/odd checking task."""
        ranges = {"easy": (1, 100), "medium": (100, 1000), "hard": (1000, 100000)}
        min_val, max_val = ranges.get(difficulty, (1, 100))
        
        n = random.randint(min_val, max_val)
        is_even = n % 2 == 0
        
        # Sometimes ask about even, sometimes odd
        if random.random() < 0.5:
            prompt = f"Is {n} an even number?"
            correct_answer = "Yes" if is_even else "No"
        else:
            prompt = f"Is {n} an odd number?"
            correct_answer = "Yes" if not is_even else "No"
        
        return {
            "prompt": prompt,
            "category": "number_property",
            "subcategory": "even_odd",
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            "metadata": {"number": n, "is_even": is_even}
        }
    
    @staticmethod
    def generate_perfect_square_task(difficulty: str) -> Dict:
        """Generate perfect square checking task."""
        if difficulty == "easy":
            # Small perfect squares
            if random.random() < 0.5:
                root = random.randint(1, 10)
                n = root * root
                is_perfect = True
            else:
                n = random.randint(2, 100)
                root = int(n ** 0.5)
                if root * root == n:
                    n += 1  # Make it not perfect
                is_perfect = False
                
        elif difficulty == "medium":
            if random.random() < 0.5:
                root = random.randint(10, 30)
                n = root * root
                is_perfect = True
            else:
                n = random.randint(100, 900)
                root = int(n ** 0.5)
                if root * root == n:
                    n += random.randint(1, 10)
                is_perfect = False
                
        else:  # hard
            if random.random() < 0.5:
                root = random.randint(30, 100)
                n = root * root
                is_perfect = True
            else:
                n = random.randint(1000, 10000)
                root = int(n ** 0.5)
                if root * root == n:
                    n += random.randint(1, 50)
                is_perfect = False
        
        prompt = f"Is {n} a perfect square?"
        correct_answer = "Yes" if is_perfect else "No"
        
        return {
            "prompt": prompt,
            "category": "number_property",
            "subcategory": "perfect_square",
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            "metadata": {"number": n, "is_perfect_square": is_perfect}
        }
    
    @staticmethod
    def generate_palindrome_task(difficulty: str) -> Dict:
        """Generate palindrome checking task."""
        if difficulty == "easy":
            # 3-digit numbers
            if random.random() < 0.5:
                # Generate palindrome
                a = random.randint(1, 9)
                b = random.randint(0, 9)
                n = int(f"{a}{b}{a}")
                is_palindrome = True
            else:
                # Generate non-palindrome
                n = random.randint(100, 999)
                s = str(n)
                if s == s[::-1]:
                    n += 1
                is_palindrome = False
                
        elif difficulty == "medium":
            # 4-5 digit numbers
            if random.random() < 0.5:
                # Generate palindrome
                if random.random() < 0.5:
                    # 4-digit
                    a, b = random.randint(1, 9), random.randint(0, 9)
                    n = int(f"{a}{b}{b}{a}")
                else:
                    # 5-digit
                    a, b, c = random.randint(1, 9), random.randint(0, 9), random.randint(0, 9)
                    n = int(f"{a}{b}{c}{b}{a}")
                is_palindrome = True
            else:
                n = random.randint(1000, 99999)
                s = str(n)
                if s == s[::-1]:
                    n += 10
                is_palindrome = False
                
        else:  # hard - words
            palindromes = ["radar", "level", "rotor", "civic", "kayak", "refer", "madam"]
            non_palindromes = ["hello", "world", "python", "binary", "dataset", "model"]
            
            if random.random() < 0.5:
                word = random.choice(palindromes)
                is_palindrome = True
            else:
                word = random.choice(non_palindromes)
                is_palindrome = False
            
            prompt = f"Is '{word}' a palindrome?"
            correct_answer = "Yes" if is_palindrome else "No"
            
            return {
                "prompt": prompt,
                "category": "pattern_recognition",
                "subcategory": "palindrome_word",
                "difficulty": difficulty,
                "correct_answer": correct_answer,
                "metadata": {"word": word, "is_palindrome": is_palindrome}
            }
        
        prompt = f"Is {n} a palindrome number?"
        correct_answer = "Yes" if is_palindrome else "No"
        
        return {
            "prompt": prompt,
            "category": "pattern_recognition",
            "subcategory": "palindrome_number",
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            "metadata": {"number": n, "is_palindrome": is_palindrome}
        }