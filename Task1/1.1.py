from itertools import combinations
import re

space = "     " # Added it globally for ease of change

class CountryStanding:
    def __init__(self, name):
        self.team = name
        self.played = 0
        self.wins = 0
        self.draw = 0
        self.losses = 0
        self.goals_scored = 0 # GF
        self.goals_against = 0 # Goals scored against us
        self.goals_difference = 0
        self.points = 0
        pass

    def update_points(self):
        self.points = 3 * self.wins + self.draw

    def update_gd(self):
        self.goals_difference = self.goals_scored - self.goals_against

    def __str__(self):
        string = ""
        for val in list(vars(self).values()):
            string += str(val) + space
        string += "\n"
        return string
    

class Cup:
    def __init__(self):
        self.standings = {
            "KSA" : CountryStanding("KSA"),
            "ARG" : CountryStanding("ARG"),
            "POL" : CountryStanding("POL"),
            "MEX" : CountryStanding("MEX")
        }


    def __str__(self):
        string = ""
        cup_space = space[1:]
        string += "Team" + cup_space + "P" + space + "W" + space + "D" + space + "L" + space + "GF" + cup_space + "GA" + cup_space +"GD" + cup_space + "Pts" + "\n"

        # Sorting results
        # We use negative as sorted sorts ascending order
        # We sort by points then gd then goals scored
        sorted_list_of_standings = sorted(
            self.standings.items(),
            key= lambda x : (
                -x[1].points,
                -x[1].goals_difference,
                -x[1].goals_scored
            )
        )
        
        for country, standing in sorted_list_of_standings:
            string += standing.__str__()
        return string
    


    def process_match(self, country1_name : str, country2_name : str, country1_score : int, country2_score : int):
        country1 = self.standings[country1_name]
        country2 = self.standings[country2_name]

        # Number of played games
        country1.played += 1
        country2.played += 1

        # Update goals for/againist
        country1.goals_scored += country1_score
        country2.goals_scored += country2_score

        country1.goals_against += country2_score
        country2.goals_against += country1_score

        # Update goal difference
        country1.update_gd()
        country2.update_gd()

        # Determine win, loss or draw
        if country1_score == country2_score:
            country1.draw += 1
            country2.draw += 1
        elif country1_score > country2_score:
            country1.wins += 1
            country2.losses += 1
        else:
            country2.wins += 1
            country1.losses += 1

        # Recalculate points for both teams
        country1.update_points()
        country2.update_points()



if __name__ == "__main__":
    cup = Cup()
    
    # Take input of matches
    countries = ["ARG", "MEX", "POL", "KSA"]
    combos = list(combinations(countries, 2))

    for country1, country2 in combos:
        while True:
            score_str = input(f"Enter score for {country1} vs {country2}: ").strip()

            # Validate input using regex
            match = re.match(r"^(\d+)-(\d+)$", score_str)

            if match:
                country1_score = int(match.group(1))
                country2_score = int(match.group(2))
                break  # Valid input, exit loop

            print("Invalid format, please reenter the scores in the format X-Y: ")


        # country1_score = int(score_str[0])
        # country2_score = int(score_str[2])
        cup.process_match(country1, country2, country1_score, country2_score)

    print(cup)