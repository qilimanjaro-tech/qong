# -------------------------------------------------------------------------

# This is a script to play a quantum version of ping pong on a quantum annealer
# For details see the README.md

# Choose the execution type, either classical simulated annealing or quantum variational annealing
execution_type = "classical"
# execution_type = "variational"

# -------------------------------------------------------------------------

import curses
from qilisdk.core import Model, BinaryVariable
from qilisdk.utils.classical_solvers import SimulatedAnnealingSolver
from qilisdk.backends import QiliSim, AnalogMethod
from qilisdk.functionals import AnalogEvolution
from qilisdk.analog import Schedule, X
from qilisdk.readout import Readout
from qilisdk.core import InitialState

# Class to hold a single data item
class DataItem:
    def __init__(self, name, value, nqubits):
        self.name = name
        self.value = value
        self.nqubits = nqubits

# This is the object that holds the game data and handles the evolution of the state
class GameData:

    # Initialize the game data with default values
    def __init__(self):
        self.data = [
            DataItem("paddle_1_y", 0, 3),
            DataItem("paddle_2_y", 0, 3),
            DataItem("ball_x", 2**4, 5),
            DataItem("ball_y", 2**2, 3),
            DataItem("ball_dir", 0, 2),
            DataItem("score_1", 0, 2),
            DataItem("score_2", 0, 2),
            DataItem("player_1_up", 0, 1),
            DataItem("player_1_down", 0, 1),
            DataItem("player_2_up", 0, 1),
            DataItem("player_2_down", 0, 1),
        ]

        # Weight for the constraints
        self.penalty_weight = 10

        # Reward for setting an indicator bit, should be less than the above
        self.reward_weight = 2

        # Simulated annealing settings
        self.num_reads = 16
        self.num_sweeps = 3000

        # Variational annealing settings
        self.anneal_time = 1000
        self.anneal_dt = 1.0
        self.anneal_order = 1
        self.anneal_shots = 50

        # Pregenerate the paramterized ising model
        self.generate_ising_model()

    # Get the value of a specific data item
    def get_value(self, name):
        for item in self.data:
            if item.name == name:
                return item.value
        return None

    # Set the value of a specific data item
    def set_value(self, name, value):
        for item in self.data:
            if item.name == name:
                item.value = value
                return
        raise ValueError(f"Data item with name '{name}' not found.")

    # Get the number of qubits required for a specific data item
    def get_nqubits(self, name):
        for item in self.data:
            if item.name == name:
                return item.nqubits
        return None

    # Get the total number of qubits required for the game data
    def total_nqubits(self):
        return sum(item.nqubits for item in self.data)

    # Get the number of qubits needing for annealing, including aux bits
    def total_nqubits_anneal(self):
        return self.total_nqubits() + len(self.aux_vars)

    # Get the number of terms in the Ising model
    def total_terms(self):
        return len(self.model.objective.term.to_list())

    # Get the bitstring index of a specific data item
    def get_bitstring_index(self, name):
        index = 0
        for item in self.data:
            if item.name == name:
                return index
            index += item.nqubits
        return None

    # Get the value of a register as an Expression
    def get_register_term(self, name, bits):
        start = self.get_bitstring_index(name)
        nqubits = self.get_nqubits(name)
        return sum(2**(nqubits-1-k) * bits[start+k] for k in range(nqubits))

    # Get the maximum value a register can hold
    def get_max_value(self, name):
        for item in self.data:
            if item.name == name:
                return 2**item.nqubits - 1
        return None

    # Convert the game data to a bitstring representation
    def to_bitstring(self):
        bitstring = ""
        for item in self.data:
            bitstring += format(item.value, f'0{item.nqubits}b')
        return bitstring

    # Convert a bitstring representation back to game data
    def from_bitstring(self, bitstring):
        index = 0
        for item in self.data:
            item.value = int(bitstring[index:index+item.nqubits], 2)
            index += item.nqubits

    # Evolve the game data one step
    def evolve(self):
        self.evolve_quantum()

    # Construct the model to solve, where x are the input bits, and y are the output bits
    def generate_ising_model(self):

        # The input and output bits
        x_vars = []
        for i in range(self.total_nqubits()):
            x_vars.append(BinaryVariable(f"x{i}"))
        y_vars = []
        for i in range(self.total_nqubits()):
            y_vars.append(BinaryVariable(f"y{i}"))

        # Build the objective
        obj_terms = []

        # The weights, kept on the object so that they are easy to tune
        penalty_weight = self.penalty_weight
        reward_weight = self.reward_weight

        # Aux bits are tracked as we go, since they need annealing alongside the output bits
        self.aux_vars = []
        def aux_var(name):
            self.aux_vars.append(BinaryVariable(name))
            return self.aux_vars[-1]

        # A Rosenberg penalty, zero only when the bit c is the AND of the binary values a and b
        def and_penalty(c, a, b):
            return penalty_weight * (3*c + a*b - 2*a*c - 2*b*c)

        # Likewise for the XOR, which needs an AND as an extra aux bit to stay quadratic
        def xor_penalty(c, a, b):
            both = aux_var(f"{c.label}_and")
            return and_penalty(both, a, b) + penalty_weight * (c - a - b + 2*both) ** 2

        # And the OR, which is the sum of the two minus their AND
        def or_penalty(c, a, b):
            both = aux_var(f"{c.label}_and")
            return and_penalty(both, a, b) + penalty_weight * (c - a - b + both) ** 2

        # A penalty, zero only when the output register represents addition by (inc - dec)
        def add_penalty(name, inc=0, dec=0, jumps=(), wrap=True):
            start = self.get_bitstring_index(name)
            nqubits = self.get_nqubits(name)
            terms = []
            carry = 0
            for k in reversed(range(nqubits)):
                addend = dec + (inc if k == nqubits - 1 else 0)
                for control, value in jumps:
                    if (value >> (nqubits-1-k)) & 1:
                        addend = addend + control
                carry_out = aux_var(f"carry_{name}_{k}") if wrap or k > 0 else dec
                terms.append(penalty_weight * (
                    x_vars[start+k] + addend + carry - y_vars[start+k] - 2*carry_out
                ) ** 2)
                carry = carry_out
            return sum(terms)

        # A penalty, zero only when the register equals the specified value
        def equality_penalty(c, name, bits, value, extra=0):
            start = self.get_bitstring_index(name)
            nqubits = self.get_nqubits(name)
            mismatch = extra + sum(
                1 - bits[start+k] if (value >> (nqubits-1-k)) & 1 else bits[start+k]
                for k in range(nqubits)
            )
            return -reward_weight * c + penalty_weight * c * mismatch

        # Which bits that we are going to change, all others should copy from input -> output
        handled_bits = []
        for name in ["paddle_1_y", "paddle_2_y", "ball_x", "ball_y", "ball_dir", "score_1", "score_2"]:
            start = self.get_bitstring_index(name)
            handled_bits += list(range(start, start + self.get_nqubits(name)))
        for i in range(self.total_nqubits()):
            if i not in handled_bits:
                obj_terms.append(penalty_weight * (x_vars[i] - y_vars[i]) ** 2)

        # Moving a paddle, up decrements its y position and down increments it
        for player in [1, 2]:
            up = x_vars[self.get_bitstring_index(f"player_{player}_up")]
            down = x_vars[self.get_bitstring_index(f"player_{player}_down")]
            obj_terms.append(add_penalty(f"paddle_{player}_y", inc=down, dec=up))

        # The ball direction bits, 0 = up right, 1 = down right, 2 = down left, 3 = up left
        dir_start = self.get_bitstring_index("ball_dir")
        dir_high = x_vars[dir_start]
        dir_low = x_vars[dir_start+1]

        # A goal is about to happen when the ball is one step from going past a paddle
        ball_x_centre = 2**(self.get_nqubits("ball_x") - 1)
        score_left = aux_var("score_left")
        score_right = aux_var("score_right")
        obj_terms.append(equality_penalty(score_left, "ball_x", x_vars, 2, extra=1 - dir_high))
        obj_terms.append(equality_penalty(score_right, "ball_x", x_vars, self.get_max_value("ball_x") - 1, extra=dir_high))

        # Whoever let it past loses, the scores wrap around if they overflow
        obj_terms.append(add_penalty("score_1", inc=score_right))
        obj_terms.append(add_penalty("score_2", inc=score_left))

        # The ball goes right when the high dir bit is clear, and down when the two dir bits differ
        moving_down = aux_var("moving_down")
        obj_terms.append(xor_penalty(moving_down, dir_high, dir_low))

        # Bits saying whether the ball is against the top or bottom wall
        at_top = aux_var("at_top")
        at_bottom = aux_var("at_bottom")
        obj_terms.append(equality_penalty(at_top, "ball_y", x_vars, 0))
        obj_terms.append(equality_penalty(at_bottom, "ball_y", x_vars, self.get_max_value("ball_y")))

        # We bounce off a wall only if we are at it and heading into it
        bounce_top = aux_var("bounce_top")
        bounce_bottom = aux_var("bounce_bottom")
        obj_terms.append(and_penalty(bounce_top, at_top, 1 - moving_down))
        obj_terms.append(and_penalty(bounce_bottom, at_bottom, moving_down))

        # A bounce cancels the vertical movement for this frame
        bounce = aux_var("bounce")
        obj_terms.append(or_penalty(bounce, bounce_top, bounce_bottom))

        # Move the ball, jumping to the center if a goal is scored
        obj_terms.append(add_penalty(
            "ball_x",
            inc=1 - dir_high - score_right,
            dec=dir_high - score_left,
            jumps=[
                (score_left, ball_x_centre - 2),
                (score_right, 2**self.get_nqubits("ball_x") + ball_x_centre - self.get_max_value("ball_x") + 1),
            ],
        ))
        obj_terms.append(add_penalty(
            "ball_y",
            inc=moving_down - bounce_bottom,
            dec=1 - moving_down - bounce_top,
            wrap=False,
        ))

        # A paddle is hit when the ball ends up heading towards it, in the column beside it and on the same row
        ball_y_start = self.get_bitstring_index("ball_y")
        hits = []
        for player, column, heading in [(1, 2, dir_high), (2, self.get_max_value("ball_x") - 1, 1 - dir_high)]:
            row_mismatch = 1 - heading
            paddle_start = self.get_bitstring_index(f"paddle_{player}_y")
            for k in range(self.get_nqubits("ball_y")):
                row_differs = aux_var(f"row_differs_{player}_{k}")
                obj_terms.append(xor_penalty(row_differs, y_vars[ball_y_start+k], y_vars[paddle_start+k]))
                row_mismatch = row_mismatch + row_differs
            hits.append(aux_var(f"paddle_hit_{player}"))
            obj_terms.append(equality_penalty(hits[-1], "ball_x", y_vars, column, extra=row_mismatch))

        # Check if either paddle is hit, or a goal was scored
        paddle_hit = aux_var("paddle_hit")
        scored = aux_var("scored")
        reverse = aux_var("reverse")
        obj_terms.append(or_penalty(paddle_hit, hits[0], hits[1]))
        obj_terms.append(or_penalty(scored, score_left, score_right))
        obj_terms.append(or_penalty(reverse, paddle_hit, scored))

        # A wall bounce flips the vertical part of the direction, which is the low dir bit
        dir_low_walled = aux_var("dir_low_walled")
        obj_terms.append(xor_penalty(dir_low_walled, dir_low, bounce))

        # Meanwhile, going back the other way flips the horizontal part (both bits)
        obj_terms.append(xor_penalty(y_vars[dir_start+1], dir_low_walled, reverse))
        obj_terms.append(xor_penalty(y_vars[dir_start], dir_high, reverse))

        # Make the model and set the objective
        model = Model("QONG")
        obj = sum(obj_terms)
        model.set_objective(obj)

        # Store the model and variables
        self.model = model.to_qubo()
        self.x_vars = x_vars
        self.y_vars = y_vars

    # Substitute the given variables with constants, returning a new model
    @staticmethod
    def partial_evaluate_model(model, input_dict):
        objective = model.objective
        new_model = Model(model.label)
        new_model.set_objective(objective.term.substitute(input_dict), sense=objective.sense)
        return new_model.to_qubo()

    # Evolve the game data one step using quantum logic
    def evolve_quantum(self):

        # Prepare the Ising model with the inputs
        to_set = self.to_bitstring()
        input_dict = {self.x_vars[i]: int(bit) for i, bit in enumerate(to_set)}
        new_model = self.partial_evaluate_model(self.model, input_dict)

        # Do the solve with classical simulated annealing
        if execution_type == "classical":
            solver = SimulatedAnnealingSolver(num_reads=self.num_reads, num_sweeps=self.num_sweeps)
            solution = solver.solve(new_model)
            new_bitstring = ''.join(str(solution.sample[self.y_vars[i]]) for i in range(len(self.y_vars)))

        # Do the solve with variational annealing
        elif execution_type == "variational":

            # Make sure the mappings are consistent
            term = new_model.objective.term
            variables = new_model.qubo_objective.variables()
            qubit_of = {v.label: i for i, v in enumerate(variables)}

            # The Hamiltonians
            final_h = new_model.to_hamiltonian() / max(abs(c) for c in term.as_coefficients_dict().values())
            init_h = -sum(X(i) for i in range(len(variables)))

            # Set up the simulation
            backend = QiliSim(analog_simulation_method=AnalogMethod.variational_annealing(order=self.anneal_order, shots=self.anneal_shots))
            schedule = Schedule.linear(init_h, final_h, self.anneal_time, self.anneal_dt)
            functional = AnalogEvolution(schedule, initial_state=InitialState.UNIFORM)
            readout = Readout().with_sampling(nshots=self.num_reads)
            solution = backend.execute(functional, readout)

            # Take the lowest energy sample
            samples = solution.get_samples()
            def sample_energy(bitstring):
                return term.evaluate({v: int(bitstring[qubit_of[v.label]]) for v in variables})
            best_sample = min(samples, key=sample_energy)
            new_bitstring = ''.join(best_sample[qubit_of[f"y{i}"]] for i in range(self.total_nqubits()))

        # Parse the results
        self.from_bitstring(new_bitstring)

# This is the object that handles the screen and user input
class GameScreen:

    def __init__(self):

        # Initialize the curses screen
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.timeout(200)
        curses.curs_set(0)

        # Create an initial game data object
        self.game_data = GameData()

    # Put the terminal back how we found it
    def cleanup(self):
        if self.screen is None:
            return
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()
        self.screen = None
        curses.curs_set(1)

    # The main game loop
    def start(self):

        # Loop forever until the user quits
        while True:

            # Check for win
            winner = 0
            if self.game_data.get_value("score_1") >= 3:
                winner = 1
            elif self.game_data.get_value("score_2") >= 3:
                winner = 2
            if winner:
                self.screen.clear()
                self.screen.addstr(3, 0, "Player {} wins! Press 'g' to quit.".format(winner))
                self.screen.refresh()
                if self.screen.getch() == ord('g'):
                    break
                continue

            # Render everything
            valid_draw = self.draw()

            # Get the key
            key = self.screen.getch()

            # Check for quit
            if key == ord('g'):
                break

            # Check for player 1 input
            if key == ord('q'):
                self.game_data.set_value("player_1_up", 1)
                self.game_data.set_value("player_1_down", 0)
            elif key == ord('a'):
                self.game_data.set_value("player_1_up", 0)
                self.game_data.set_value("player_1_down", 1)
            else:
                self.game_data.set_value("player_1_up", 0)
                self.game_data.set_value("player_1_down", 0)

            # Check for player 2 input
            if key == ord('o'):
                self.game_data.set_value("player_2_up", 1)
                self.game_data.set_value("player_2_down", 0)
            elif key == ord('l'):
                self.game_data.set_value("player_2_up", 0)
                self.game_data.set_value("player_2_down", 1)
            else:
                self.game_data.set_value("player_2_up", 0)
                self.game_data.set_value("player_2_down", 0)

            # Evolve the game data one step
            if valid_draw:
                self.game_data.evolve()

    # Draw the game screen
    def draw(self):

        # Start with a fresh screen
        self.screen.clear()

        # Some helpful text
        self.screen.addstr(0, 0, "QONG: 'g' to quit, 'q'/'a' to move player 1, 'o'/'l' to move player 2")
        self.screen.addstr(1, 0, "Score: {} - {}   (first to 3)".format(self.game_data.get_value("score_1"), self.game_data.get_value("score_2")))
        self.screen.addstr(2, 0, "State: |{}>, game qubits: {}, total qubits: {}, terms: {}".format(self.game_data.to_bitstring(), self.game_data.total_nqubits(), self.game_data.total_nqubits_anneal(), self.game_data.total_terms()))

        # Get parameters for drawing the game
        game_starts_at_line = 5
        screen_height = self.screen.getmaxyx()[0]
        screen_width = self.screen.getmaxyx()[1]
        game_height = self.game_data.get_max_value("ball_y") + 1
        game_width = self.game_data.get_max_value("ball_x") + 1
        max_x = game_width - 1

        # Make sure we can fit the game on the screen
        if game_starts_at_line + game_height > screen_height or game_width > screen_width:
            self.screen.addstr(game_starts_at_line, 0, "Screen too small to draw the game!")
            self.screen.refresh()
            return False

        # Draw the arena
        for y in range(game_height):
            self.screen.addstr(y + game_starts_at_line, 0, "|")
            self.screen.addstr(y + game_starts_at_line, game_width, "|")
        for x in range(game_width + 1):
            self.screen.addstr(game_starts_at_line - 1, x, "-")
            self.screen.addstr(game_starts_at_line + game_height, x, "-")
        self.screen.addstr(game_starts_at_line - 1, 1, "^")
        self.screen.addstr(game_starts_at_line + game_height, 1, "v")
        self.screen.addstr(game_starts_at_line - 1, game_width - 1, "^")
        self.screen.addstr(game_starts_at_line + game_height, game_width - 1, "v")

        # Draw the paddles
        paddle_1_y = self.game_data.get_value("paddle_1_y")
        paddle_2_y = self.game_data.get_value("paddle_2_y")
        self.screen.addstr(paddle_1_y + game_starts_at_line, 1, "]")
        self.screen.addstr(paddle_2_y + game_starts_at_line, max_x, "[")

        # Draw the ball
        ball_x = self.game_data.get_value("ball_x")
        ball_y = self.game_data.get_value("ball_y")
        self.screen.addstr(ball_y + game_starts_at_line, ball_x, "O")

        # Show it
        self.screen.refresh()

        # Everything rendered, so we should evolve the state
        return True

# Start it up
if __name__ == "__main__":
    screen = GameScreen()
    try:
        screen.start()
    except KeyboardInterrupt:
        pass
    finally:
        screen.cleanup()
