from enum import Enum, auto
from typing import List
import random
import uuid
from dataclasses import dataclass # class decorators

# We have to give a value for each entry in an enum
# But since we don't care about this value, we use auto() which gives values
# without exposing it to us since we don't care
class Position(Enum):
    FORWARD = auto()
    MIDFIELDER = auto()
    DEFENDER = auto()
    GOALKEEPER = auto()

class EventType(Enum):
    GOAL = auto()
    SUBSTITUTION = auto()
    HALF_TIME = auto()
    FULL_TIME = auto()

class MatchPhase(Enum):
    REGULATION = auto()
    FINISHED = auto()


class Player:
    def __init__(self, name : str, position, base_attack, base_defense):
        self.name = name
        self.position = position
        self.base_attack = base_attack
        self.base_defense = base_defense
        self.stamina = 100 # Starts at 100

    def deplete_stamina(self, rate):
        self.stamina = max(10.0, self.stamina - rate)

    def get_effective_attack(self):
        return self.base_attack * (self.stamina / 100)

    def get_effective_defense(self):
        return self.base_defense * (self.stamina / 100)



class Team:
    def __init__(self, country_name : str, roster : List[Player], active_lineup : List[Player], bench : List[Player]):
        self.country_name = country_name
        self.roster = roster
        self.active_lineup = active_lineup
        self.substitutions_remaining : int = 5


    @property
    def bench(self):
        # bench = roster minus active_lineup
        return [player for player in self.roster if player not in self.active_lineup]

    def get_aggregate_attack(self):
        attackers = [p for p in self.active_lineup if p.position in (Position.FORWARD, Position.MIDFIELDER)]
        if not attackers: return 0.0
        return sum(p.get_effective_attack() for p in attackers) / len(attackers)


    def get_aggregate_defense(self):
        defenders = [p for p in self.active_lineup if p.position in (Position.DEFENDER, Position.GOALKEEPER)]
        if not defenders: return 0.0
        return sum(p.get_effective_defense() for p in defenders) / len(defenders)


    def execute_substitution(self, player_out, player_in):
        if self.substitutions_remaining > 0 and player_out in self.active_lineup and player_in in self.bench:
            self.active_lineup.remove(player_out)
            self.active_lineup.append(player_in)
            self.substitutions_remaining -= 1
            return True
        return False


@dataclass(frozen = True) # Marks it as immutable
class MatchEvent:

    # For immutable classes, attrs are specified like that
    # and u have to specify them all when constructing the class
    # since it's immutable
    event_id : str = str(uuid.uuid4()) # assign a unique id to each event
    event_type : EventType
    minute : int
    team : Team
    player : Player
    outcome_text : str

    def to_string(self):
        return f"[{self.minute}] {self.event_type.name} | {self.team.country_name} | {self.player.name} -> {self.outcome_text}"


class Match:
    def __init__(self, home_team: Team, away_team: Team):
            self.home_team = home_team
            self.away_team = away_team
            self.home_score: int = 0
            self.away_score: int = 0
            self.current_minute: int = 0
            self.timeline: List[MatchEvent] = []
            self.phase: MatchPhase = MatchPhase.REGULATION
            
            self.base_decay: float = 0.5


    def run_minute_tick(self):
        if self.phase == MatchPhase.FINISHED:
            return

        self.current_minute += 1

        # do stamina decay
        for player in self.home_team.active_lineup + self.away_team.active_lineup:
            player.deplete_stamina(self.base_decay)

        self.process_goal_attempt(self.home_team, self.away_team)
        self.process_goal_attempt(self.away_team, self.home_team)

        if self.current_minute == 90:
            self.phase = MatchPhase.FINISHED

            result_text = "DRAW"
            if self.home_score > self.away_score:
                result_text = f"{self.home_team.country_name} WINS"
            elif self.away_score > self.home_score:
                result_text = f"{self.away_team.country_name} WINS"

            self.timeline.append(MatchEvent(
                event_type=EventType.FULL_TIME,
                minute=self.current_minute,
                team=None,
                player=None,
                outcome_text= result_text
            ))

    def process_goal_attempt(self, attacking_team : Team, defending_team : Team):

        if random.Random() < 0.10:
            aggregate_attack = attacking_team.get_aggregate_attack()
            aggregate_defense = defending_team.get_aggregate_defense()

            attack_value = aggregate_attack * random.uniform(0.75, 1.25)
            defense_value = aggregate_defense * 1.3 * random.uniform(0.80, 1.20)

            if attack_value > defense_value:
                # Goal Scored😎
                scoring_player = random.choice([p for p in attacking_team.active_lineup 
                                                if p.position in (Position.FORWARD, Position.MIDFIELDER)])
                
                # add GOAL event
                event = MatchEvent(
                    event_type=EventType.GOAL,
                    minute=self.current_minute,
                    team=attacking_team,
                    player=scoring_player,
                    outcome_text=f"GOAL by {attacking_team}"
                )
                self.timeline.append(event)
                
                # Update Score
                if attacking_team == self.home_team:
                    self.home_score += 1
                else:
                    self.away_score += 1

# AI helped me on this fn! (I am not much into football)
def create_test_team(country_name : str) -> Team:
    roster = []
    active_lineup = []

    # 4-4-2
    active_positions = [
        Position.GOALKEEPER,
        Position.DEFENDER, Position.DEFENDER, Position.DEFENDER, Position.DEFENDER,
        Position.MIDFIELDER, Position.MIDFIELDER, Position.MIDFIELDER, Position.MIDFIELDER,
        Position.FORWARD, Position.FORWARD
    ]

    for i, pos in enumerate(active_positions):
        # Assign random base attack/defense stats between 70 and 95
        player = Player(
            name=f"{country_name}_Player_{i+1}", 
            position=pos, 
            base_attack=random.randint(70, 95), 
            base_defense=random.randint(70, 95)
        )
        active_lineup.append(player)
        roster.append(player)


    # Generate 15 bench players
    for i in range(11, 26):
        # Bench players get slightly lower random stats
        bench_player = Player(
            name=f"{country_name}_Sub_{i+1}", 
            position=random.choice(list(Position)), 
            base_attack=random.randint(60, 85), 
            base_defense=random.randint(60, 85)
        )
        roster.append(bench_player)
        
    return Team(country_name, roster, active_lineup)


def run_simulation():
    # Instantiate two rival teams
    home_team = create_test_team("ARG")
    away_team = create_test_team("FRA")
    
    # Initialize the match engine
    championship_match = Match(home_team, away_team)
    
    print(f"=====================================================")
    print(f"⚽ KICKOFF: {home_team.country_name} vs {away_team.country_name}")
    print(f"=====================================================")
    
    # Run the Regulation Phase simulation loop (Minutes 1 to 90)
    while championship_match.phase == MatchPhase.REGULATION:
        championship_match.run_minute_tick()
        
        # Optional: Print a half-time marker for visual clarity
        if championship_match.current_minute == 45:
            print("--- ⏱️ HALF TIME ---")
            
    # Output the definitive timeline history
    print("\n--- 📜 MATCH TIMELINE ---")
    if not championship_match.timeline:
         print("No major events occurred.")
    else:
        for event in championship_match.timeline:
            print(event.to_string())
            
    # Display the final results
    print(f"\n=====================================================")
    print(f"🏆 FINAL SCORE: {home_team.country_name} {championship_match.home_score} - {championship_match.away_score} {away_team.country_name}")
    print(f"=====================================================")

if __name__ == "__main__":
    run_simulation()