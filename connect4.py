import numpy as np
import random
from typing import List, Tuple

class Connect4:
    """Connect 4 game environment"""
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        
    def reset(self, randomize_start: bool = True):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        # Randomize who goes first for balanced training
        if randomize_start:
            self.current_player = random.choice([1, 2])
        else:
            self.current_player = 1
        return self.board.copy()
    
    def get_valid_moves(self) -> List[int]:
        return [col for col in range(self.cols) if self.board[0][col] == 0]
    
    def make_move(self, col: int) -> Tuple[np.ndarray, float, bool]:
        """Make a move and return (new_state, reward, done)"""
        if col not in self.get_valid_moves():
            return self.board.copy(), -10, True
        
        # Drop piece
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break
        
        # Check win
        if self._check_win(self.current_player):
            return self.board.copy(), 1, True
        
        # Check draw
        if len(self.get_valid_moves()) == 0:
            return self.board.copy(), 0, True
        
        # Switch player
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0, False
    
    def _check_win(self, player: int) -> bool:
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row][col+i] == player for i in range(4)):
                    return True
        
        # Vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row+i][col] == player for i in range(4)):
                    return True
        
        # Diagonal (down-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row+i][col+i] == player for i in range(4)):
                    return True
        
        # Diagonal (down-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(self.board[row+i][col-i] == player for i in range(4)):
                    return True
        
        return False


class NeuralNetwork:
    """Simple feedforward neural network"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.01):
        self.lr = learning_rate
        
        # He initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """Forward pass"""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.z2
        return self.a2
    
    def backward(self, x, target, output):
        """Backward pass using MSE loss"""
        m = x.shape[0]
        
        # Output layer gradients
        dz2 = (output - target) / m
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
    
    def predict(self, x):
        return self.forward(x)


class Connect4Agent:
    """RL Agent for Connect 4"""
    def __init__(self, epsilon: float = 0.3, gamma: float = 0.95):
        self.nn = NeuralNetwork(42, 64, 7, learning_rate=0.001)
        self.epsilon = epsilon
        self.gamma = gamma
        
    def flip_state(self, state: np.ndarray) -> np.ndarray:
        """Flip the board perspective (swap 1s and 2s)"""
        flipped = state.copy()
        flipped[state == 1] = 2
        flipped[state == 2] = 1
        return flipped
    
    def get_action(self, state: np.ndarray, valid_moves: List[int], training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        state_flat = state.flatten().reshape(1, -1)
        q_values = self.nn.predict(state_flat)[0]
        
        # Mask invalid moves
        masked_q = np.full(7, -np.inf)
        for move in valid_moves:
            masked_q[move] = q_values[move]
        
        return np.argmax(masked_q)
    
    def train_step(self, state: np.ndarray, action: int, reward: float, 
                   next_state: np.ndarray, done: bool, valid_next_moves: List[int]):
        """Single training step using Q-learning"""
        state_flat = state.flatten().reshape(1, -1)
        next_state_flat = next_state.flatten().reshape(1, -1)
        
        # Current Q-values
        current_q = self.nn.predict(state_flat)
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            next_q = self.nn.predict(next_state_flat)[0]
            masked_next_q = np.full(7, -np.inf)
            for move in valid_next_moves:
                masked_next_q[move] = next_q[move]
            target_q = reward + self.gamma * np.max(masked_next_q)
        
        # Update only the action taken
        target = current_q.copy()
        target[0][action] = target_q
        
        # Backpropagation
        self.nn.backward(state_flat, target, current_q)


def train_selfplay(episodes: int = 2000, verbose: bool = True):
    """
    Train agent through SELF-PLAY
    Both players use the same agent, so it learns from both perspectives
    """
    agent = Connect4Agent(epsilon=0.3)
    env = Connect4()
    
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    for episode in range(episodes):
        state = env.reset(randomize_start=True)  # Randomize who starts!
        done = False
        
        # Store experiences for both players
        p1_history = []  # Player 1's experiences
        p2_history = []  # Player 2's experiences
        
        move_count = 0
        starting_player = env.current_player  # Track who started
        
        while not done:
            current_player = env.current_player
            valid_moves = env.get_valid_moves()
            
            # Get action from agent (same agent for both players!)
            if current_player == 1:
                # Player 1 sees the board as-is
                action = agent.get_action(state, valid_moves)
                next_state, reward, done = env.make_move(action)
                p1_history.append((state.copy(), action, reward, next_state.copy(), done))
            else:
                # Player 2 sees the board FLIPPED (1s become 2s, 2s become 1s)
                # This way the agent always learns from "my pieces vs opponent pieces"
                flipped_state = agent.flip_state(state)
                action = agent.get_action(flipped_state, valid_moves)
                next_state, reward, done = env.make_move(action)
                flipped_next = agent.flip_state(next_state)
                p2_history.append((flipped_state, action, reward, flipped_next, done))
            
            state = next_state
            move_count += 1
            
            # Safety: prevent infinite loops
            if move_count > 100:
                done = True
                reward = 0
        
        # Determine final rewards from each player's perspective
        if reward == 1:  # Last player won
            if env.current_player == 2:  # Player 1 won (switched after move)
                p1_wins += 1
                final_reward_p1 = 1
                final_reward_p2 = -1
            else:  # Player 2 won
                p2_wins += 1
                final_reward_p1 = -1
                final_reward_p2 = 1
        else:  # Draw
            draws += 1
            final_reward_p1 = 0
            final_reward_p2 = 0
        
        # Update final rewards in histories
        if p1_history:
            p1_history[-1] = (p1_history[-1][0], p1_history[-1][1], 
                             final_reward_p1, p1_history[-1][3], True)
        if p2_history:
            p2_history[-1] = (p2_history[-1][0], p2_history[-1][1], 
                             final_reward_p2, p2_history[-1][3], True)
        
        # Propagate rewards backwards (opponent's moves get negative reward)
        for i in range(len(p1_history) - 2, -1, -1):
            state, action, _, next_state, _ = p1_history[i]
            # If opponent won next, this move gets penalized
            if p1_history[i + 1][2] < 0:  # Next move led to opponent win
                p1_history[i] = (state, action, -0.1, next_state, False)
        
        for i in range(len(p2_history) - 2, -1, -1):
            state, action, _, next_state, _ = p2_history[i]
            if p2_history[i + 1][2] < 0:
                p2_history[i] = (state, action, -0.1, next_state, False)
        
        # Train on BOTH players' experiences
        for state, action, reward, next_state, done in p1_history:
            valid_next = [c for c in range(7) if next_state[0][c] == 0]
            agent.train_step(state, action, reward, next_state, done, valid_next)
        
        for state, action, reward, next_state, done in p2_history:
            valid_next = [c for c in range(7) if next_state[0][c] == 0]
            agent.train_step(state, action, reward, next_state, done, valid_next)
        
        # Decay epsilon
        agent.epsilon = max(0.05, agent.epsilon * 0.995)
        
        if verbose and (episode + 1) % 100 == 0:
            total = episode + 1
            print(f"Episode {total}/{episodes}")
            print(f"  P1 Wins: {p1_wins} ({p1_wins/total*100:.1f}%)")
            print(f"  P2 Wins: {p2_wins} ({p2_wins/total*100:.1f}%)")
            print(f"  Draws: {draws} ({draws/total*100:.1f}%)")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg moves per game: {(len(p1_history) + len(p2_history)):.1f}")
            print(f"  (Games randomize starting player for balanced learning)")
            print()
    
    return agent


def play_game(agent: Connect4Agent):
    """Play a game against the trained agent"""
    env = Connect4()
    state = env.reset()
    
    print("\nConnect 4 Game Started!")
    print("You are Player 2 (represented by O)")
    print("Columns are numbered 0-6")
    print_board(state)
    
    while True:
        # Agent's turn (Player 1)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        action = agent.get_action(state, valid_moves, training=False)
        print(f"\nAgent plays column {action}")
        state, reward, done = env.make_move(action)
        print_board(state)
        
        if done:
            if reward == 1:
                print("Agent wins!")
            else:
                print("Draw!")
            break
        
        # Human's turn (Player 2)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        while True:
            try:
                col = int(input(f"\nYour turn! Choose column (0-6): "))
                if col in valid_moves:
                    break
                print(f"Invalid move! Valid columns: {valid_moves}")
            except ValueError:
                print("Please enter a number")
        
        state, reward, done = env.make_move(col)
        print_board(state)
        
        if done:
            if reward == 1:
                print("You win!")
            else:
                print("Draw!")
            break


def print_board(board: np.ndarray):
    """Print the board"""
    print("\n" + "=" * 29)
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print(" . ", end="|")
            elif cell == 1:
                print(" X ", end="|")
            else:
                print(" O ", end="|")
        print()
    print("=" * 29)
    print("  0   1   2   3   4   5   6")


def human_teaching_session(agent: Connect4Agent):
    """Allow human to demonstrate good moves to teach the agent"""
    env = Connect4()
    state = env.reset(randomize_start=False)  # Always start fresh
    
    print("\n" + "="*60)
    print("HUMAN TEACHING MODE")
    print("="*60)
    print("\nYou can demonstrate good moves to help train the agent!")
    print("Commands:")
    print("  - Enter column number (0-6) to make a move")
    print("  - Type 'teach X' to demonstrate that column X is the best move")
    print("  - Type 'reset' to start a new board")
    print("  - Type 'done' to finish teaching")
    print("="*60)
    
    teaching_examples = []
    
    while True:
        print_board(state)
        
        # Show what the agent currently thinks
        valid_moves = env.get_valid_moves()
        state_flat = state.flatten().reshape(1, -1)
        q_values = agent.nn.predict(state_flat)[0]
        
        print("\nAgent's current Q-values for valid moves:")
        for col in valid_moves:
            print(f"  Column {col}: {q_values[col]:.3f}")
        
        agent_choice = valid_moves[np.argmax([q_values[col] for col in valid_moves])]
        print(f"\nAgent would choose: Column {agent_choice}")
        
        # Get human input
        command = input("\nYour command (column/teach X/reset/done): ").strip().lower()
        
        if command == 'done':
            break
        
        if command == 'reset':
            state = env.reset(randomize_start=False)
            print("\nBoard reset!")
            continue
        
        # Teaching command: "teach X"
        if command.startswith('teach'):
            parts = command.split()
            if len(parts) == 2:
                try:
                    best_col = int(parts[1])
                    if best_col in valid_moves:
                        # Save this as a teaching example
                        teaching_examples.append((state.copy(), best_col))
                        print(f"\n✓ Recorded! You taught: Column {best_col} is best here")
                        print(f"  (Agent thought Column {agent_choice} was best)")
                        
                        # Immediately train on this example
                        # Set high Q-value for the taught move
                        state_flat = state.flatten().reshape(1, -1)
                        current_q = agent.nn.predict(state_flat)
                        target = current_q.copy()
                        
                        # Give the taught move a high target Q-value
                        target[0][best_col] = 1.0
                        
                        # Give other moves lower values (especially the wrong choice)
                        for col in valid_moves:
                            if col != best_col:
                                target[0][col] = max(-0.5, current_q[0][col] - 0.3)
                        
                        # Train multiple times on this example for emphasis
                        for _ in range(5):
                            agent.nn.backward(state_flat, target, current_q)
                            current_q = agent.nn.predict(state_flat)
                        
                        print("  Agent has been updated with this knowledge!")
                        continue
                    else:
                        print(f"Column {best_col} is not valid! Valid: {valid_moves}")
                        continue
                except ValueError:
                    print("Invalid format. Use: teach 3")
                    continue
            else:
                print("Invalid format. Use: teach 3")
                continue
        
        # Regular move
        try:
            col = int(command)
            if col not in valid_moves:
                print(f"Invalid move! Valid columns: {valid_moves}")
                continue
            
            # Make the move
            state, reward, done = env.make_move(col)
            
            if done:
                print_board(state)
                if reward == 1:
                    print(f"\nPlayer {3 - env.current_player} wins!")
                else:
                    print("\nDraw!")
                state = env.reset(randomize_start=False)
                print("\nStarting new board...")
            
        except ValueError:
            print("Invalid command. Try: 0-6, teach X, reset, or done")
    
    print(f"\n✓ Teaching session complete! You provided {len(teaching_examples)} examples.")
    return teaching_examples


def play_with_hints(agent: Connect4Agent):
    """Play against agent with option to see/teach during game"""
    env = Connect4()
    state = env.reset()
    
    print("\n" + "="*60)
    print("PLAY WITH TEACHING MODE")
    print("="*60)
    print("You are Player 2 (O)")
    print("Commands:")
    print("  - Enter column (0-6) to play")
    print("  - Type 'hint' to see agent's thinking")
    print("  - Type 'teach X' when agent plays badly to correct it")
    print("="*60)
    
    print_board(state)
    
    while True:
        # Agent's turn (Player 1)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        action = agent.get_action(state, valid_moves, training=False)
        print(f"\nAgent plays column {action}")
        
        agent_state = state.copy()
        state, reward, done = env.make_move(action)
        print_board(state)
        
        if done:
            if reward == 1:
                print("Agent wins!")
            else:
                print("Draw!")
            break
        
        # Ask if human wants to teach about that move
        teach_input = input("\nWas that move bad? Type 'teach X' to show better move, or press ENTER: ").strip().lower()
        if teach_input.startswith('teach'):
            parts = teach_input.split()
            if len(parts) == 2:
                try:
                    better_col = int(parts[1])
                    # Undo the agent's move
                    env.board = agent_state
                    env.current_player = 1
                    state = agent_state
                    
                    valid_undo = env.get_valid_moves()
                    if better_col in valid_undo:
                        # Train agent that better_col was better
                        state_flat = state.flatten().reshape(1, -1)
                        current_q = agent.nn.predict(state_flat)
                        target = current_q.copy()
                        target[0][better_col] = 1.0
                        target[0][action] = -0.5  # Penalize the bad move
                        
                        for _ in range(10):
                            agent.nn.backward(state_flat, target, current_q)
                            current_q = agent.nn.predict(state_flat)
                        
                        print(f"✓ Agent learned! Column {better_col} is better than {action}")
                        
                        # Make the better move
                        state, reward, done = env.make_move(better_col)
                        print(f"\nReplaying with column {better_col}:")
                        print_board(state)
                        
                        if done:
                            if reward == 1:
                                print("Agent wins!")
                            else:
                                print("Draw!")
                            break
                except ValueError:
                    pass
        
        # Human's turn (Player 2)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        while True:
            move_input = input(f"\nYour turn! (0-6, or 'hint'): ").strip().lower()
            
            if move_input == 'hint':
                # Show agent's evaluation of the position (from Player 2's perspective)
                flipped = agent.flip_state(state)
                state_flat = flipped.flatten().reshape(1, -1)
                q_values = agent.nn.predict(state_flat)[0]
                
                print("\nAgent's Q-values for YOUR position:")
                for col in valid_moves:
                    print(f"  Column {col}: {q_values[col]:.3f}")
                best = valid_moves[np.argmax([q_values[col] for col in valid_moves])]
                print(f"Agent suggests: Column {best}")
                continue
            
            try:
                col = int(move_input)
                if col in valid_moves:
                    break
                print(f"Invalid! Valid: {valid_moves}")
            except ValueError:
                print("Enter a number 0-6 or 'hint'")
        
        state, reward, done = env.make_move(col)
        print_board(state)
        
        if done:
            if reward == 1:
                print("You win!")
            else:
                print("Draw!")
            break


def get_training_parameters():
    """Interactive parameter configuration"""
    print("="*60)
    print("CONNECT 4 NEURAL NETWORK TRAINING CONFIGURATOR")
    print("="*60)
    print("\nPress ENTER to use default values shown in [brackets]\n")
    
    # Episodes
    while True:
        try:
            episodes_input = input("Number of training episodes [2000]: ").strip()
            episodes = int(episodes_input) if episodes_input else 2000
            if episodes > 0:
                break
            print("Must be positive!")
        except ValueError:
            print("Please enter a valid number")
    
    # Learning rate
    while True:
        try:
            lr_input = input("Learning rate (0.0001 to 0.1) [0.001]: ").strip()
            learning_rate = float(lr_input) if lr_input else 0.001
            if 0.0001 <= learning_rate <= 0.1:
                break
            print("Must be between 0.0001 and 0.1")
        except ValueError:
            print("Please enter a valid number")
    
    # Epsilon (exploration rate)
    while True:
        try:
            eps_input = input("Starting epsilon/exploration (0.0 to 1.0) [0.3]: ").strip()
            epsilon = float(eps_input) if eps_input else 0.3
            if 0.0 <= epsilon <= 1.0:
                break
            print("Must be between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    # Gamma (discount factor)
    while True:
        try:
            gamma_input = input("Gamma/discount factor (0.0 to 1.0) [0.95]: ").strip()
            gamma = float(gamma_input) if gamma_input else 0.95
            if 0.0 <= gamma <= 1.0:
                break
            print("Must be between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    # Hidden layer size
    while True:
        try:
            hidden_input = input("Hidden layer neurons (16 to 256) [64]: ").strip()
            hidden_size = int(hidden_input) if hidden_input else 64
            if 16 <= hidden_size <= 256:
                break
            print("Must be between 16 and 256")
        except ValueError:
            print("Please enter a valid number")
    
    # Epsilon decay
    while True:
        try:
            decay_input = input("Epsilon decay rate (0.9 to 0.999) [0.995]: ").strip()
            epsilon_decay = float(decay_input) if decay_input else 0.995
            if 0.9 <= epsilon_decay <= 0.999:
                break
            print("Must be between 0.9 and 0.999")
        except ValueError:
            print("Please enter a valid number")
    
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY:")
    print("="*60)
    print(f"Episodes:              {episodes}")
    print(f"Learning Rate:         {learning_rate}")
    print(f"Starting Epsilon:      {epsilon}")
    print(f"Gamma (Discount):      {gamma}")
    print(f"Hidden Layer Size:     {hidden_size}")
    print(f"Epsilon Decay:         {epsilon_decay}")
    print("="*60)
    
    confirm = input("\nStart training with these parameters? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled.")
        return None
    
    return {
        'episodes': episodes,
        'learning_rate': learning_rate,
        'epsilon': epsilon,
        'gamma': gamma,
        'hidden_size': hidden_size,
        'epsilon_decay': epsilon_decay
    }


if __name__ == "__main__":
    print("="*60)
    print("SELF-PLAY TRAINING")
    print("="*60)
    print("The agent plays BOTH sides and learns from both perspectives!")
    print("This dramatically improves learning speed and final strength.\n")
    
    # Get parameters from user
    params = get_training_parameters()
    
    if params is None:
        exit()
    
    # Create agent with custom parameters
    agent = Connect4Agent(epsilon=params['epsilon'], gamma=params['gamma'])
    agent.nn = NeuralNetwork(42, params['hidden_size'], 7, 
                            learning_rate=params['learning_rate'])
    
    # Store epsilon decay for use in training
    epsilon_decay = params['epsilon_decay']
    
    # Modified training loop with custom epsilon decay
    env = Connect4()
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    print("\n" + "="*60)
    print("TRAINING IN PROGRESS...")
    print("="*60 + "\n")
    
    for episode in range(params['episodes']):
        state = env.reset(randomize_start=True)
        done = False
        
        p1_history = []
        p2_history = []
        move_count = 0
        starting_player = env.current_player
        
        while not done:
            current_player = env.current_player
            valid_moves = env.get_valid_moves()
            
            if current_player == 1:
                action = agent.get_action(state, valid_moves)
                next_state, reward, done = env.make_move(action)
                p1_history.append((state.copy(), action, reward, next_state.copy(), done))
            else:
                flipped_state = agent.flip_state(state)
                action = agent.get_action(flipped_state, valid_moves)
                next_state, reward, done = env.make_move(action)
                flipped_next = agent.flip_state(next_state)
                p2_history.append((flipped_state, action, reward, flipped_next, done))
            
            state = next_state
            move_count += 1
            
            if move_count > 100:
                done = True
                reward = 0
        
        # Determine final rewards
        if reward == 1:
            if env.current_player == 2:
                p1_wins += 1
                final_reward_p1 = 1
                final_reward_p2 = -1
            else:
                p2_wins += 1
                final_reward_p1 = -1
                final_reward_p2 = 1
        else:
            draws += 1
            final_reward_p1 = 0
            final_reward_p2 = 0
        
        # Update final rewards
        if p1_history:
            p1_history[-1] = (p1_history[-1][0], p1_history[-1][1], 
                             final_reward_p1, p1_history[-1][3], True)
        if p2_history:
            p2_history[-1] = (p2_history[-1][0], p2_history[-1][1], 
                             final_reward_p2, p2_history[-1][3], True)
        
        # Propagate rewards
        for i in range(len(p1_history) - 2, -1, -1):
            state_h, action_h, _, next_state_h, _ = p1_history[i]
            if p1_history[i + 1][2] < 0:
                p1_history[i] = (state_h, action_h, -0.1, next_state_h, False)
        
        for i in range(len(p2_history) - 2, -1, -1):
            state_h, action_h, _, next_state_h, _ = p2_history[i]
            if p2_history[i + 1][2] < 0:
                p2_history[i] = (state_h, action_h, -0.1, next_state_h, False)
        
        # Train on both players
        for state_t, action_t, reward_t, next_state_t, done_t in p1_history:
            valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
            agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
        
        for state_t, action_t, reward_t, next_state_t, done_t in p2_history:
            valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
            agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
        
        # Decay epsilon with custom rate
        agent.epsilon = max(0.05, agent.epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            total = episode + 1
            print(f"Episode {total}/{params['episodes']}")
            print(f"  P1 Wins: {p1_wins} ({p1_wins/total*100:.1f}%)")
            print(f"  P2 Wins: {p2_wins} ({p2_wins/total*100:.1f}%)")
            print(f"  Draws: {draws} ({draws/total*100:.1f}%)")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print()
    
    trained_agent = agent
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Statistics:")
    print(f"  Total Games: {params['episodes']}")
    print(f"  P1 Win Rate: {p1_wins/params['episodes']*100:.1f}%")
    print(f"  P2 Win Rate: {p2_wins/params['episodes']*100:.1f}%")
    print(f"  Draw Rate: {draws/params['episodes']*100:.1f}%")
    print(f"  Final Epsilon: {agent.epsilon:.3f}")
    print("="*60)
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Play against the agent")
        print("2. Play with teaching mode (correct agent during game)")
        print("3. Teaching session (show agent good positions)")
        print("4. Train more episodes")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            play_game(trained_agent)
        elif choice == '2':
            play_with_hints(trained_agent)
        elif choice == '3':
            human_teaching_session(trained_agent)
        elif choice == '4':
            # Additional training
            while True:
                try:
                    extra_episodes = int(input("How many more episodes? "))
                    if extra_episodes > 0:
                        break
                except ValueError:
                    pass
            
            print(f"\nTraining {extra_episodes} more episodes...")
            
            # Continue training with same agent
            for episode in range(extra_episodes):
                state = env.reset(randomize_start=True)
                done = False
                
                p1_history = []
                p2_history = []
                move_count = 0
                
                while not done:
                    current_player = env.current_player
                    valid_moves = env.get_valid_moves()
                    
                    if current_player == 1:
                        action = trained_agent.get_action(state, valid_moves)
                        next_state, reward, done = env.make_move(action)
                        p1_history.append((state.copy(), action, reward, next_state.copy(), done))
                    else:
                        flipped_state = trained_agent.flip_state(state)
                        action = trained_agent.get_action(flipped_state, valid_moves)
                        next_state, reward, done = env.make_move(action)
                        flipped_next = trained_agent.flip_state(next_state)
                        p2_history.append((flipped_state, action, reward, flipped_next, done))
                    
                    state = next_state
                    move_count += 1
                    
                    if move_count > 100:
                        done = True
                        reward = 0
                
                if reward == 1:
                    if env.current_player == 2:
                        final_reward_p1 = 1
                        final_reward_p2 = -1
                    else:
                        final_reward_p1 = -1
                        final_reward_p2 = 1
                else:
                    final_reward_p1 = 0
                    final_reward_p2 = 0
                
                if p1_history:
                    p1_history[-1] = (p1_history[-1][0], p1_history[-1][1], 
                                     final_reward_p1, p1_history[-1][3], True)
                if p2_history:
                    p2_history[-1] = (p2_history[-1][0], p2_history[-1][1], 
                                     final_reward_p2, p2_history[-1][3], True)
                
                for i in range(len(p1_history) - 2, -1, -1):
                    state_h, action_h, _, next_state_h, _ = p1_history[i]
                    if p1_history[i + 1][2] < 0:
                        p1_history[i] = (state_h, action_h, -0.1, next_state_h, False)
                
                for i in range(len(p2_history) - 2, -1, -1):
                    state_h, action_h, _, next_state_h, _ = p2_history[i]
                    if p2_history[i + 1][2] < 0:
                        p2_history[i] = (state_h, action_h, -0.1, next_state_h, False)
                
                for state_t, action_t, reward_t, next_state_t, done_t in p1_history:
                    valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
                    trained_agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
                
                for state_t, action_t, reward_t, next_state_t, done_t in p2_history:
                    valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
                    trained_agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
                
                trained_agent.epsilon = max(0.05, trained_agent.epsilon * epsilon_decay)
            
            print(f"✓ Trained {extra_episodes} more episodes!")
            
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice!")import numpy as np
import random
from typing import List, Tuple

class Connect4:
    """Connect 4 game environment"""
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        
    def reset(self, randomize_start: bool = True):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        # Randomize who goes first for balanced training
        if randomize_start:
            self.current_player = random.choice([1, 2])
        else:
            self.current_player = 1
        return self.board.copy()
    
    def get_valid_moves(self) -> List[int]:
        return [col for col in range(self.cols) if self.board[0][col] == 0]
    
    def make_move(self, col: int) -> Tuple[np.ndarray, float, bool]:
        """Make a move and return (new_state, reward, done)"""
        if col not in self.get_valid_moves():
            return self.board.copy(), -10, True
        
        # Drop piece
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break
        
        # Check win
        if self._check_win(self.current_player):
            return self.board.copy(), 1, True
        
        # Check draw
        if len(self.get_valid_moves()) == 0:
            return self.board.copy(), 0, True
        
        # Switch player
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0, False
    
    def _check_win(self, player: int) -> bool:
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row][col+i] == player for i in range(4)):
                    return True
        
        # Vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row+i][col] == player for i in range(4)):
                    return True
        
        # Diagonal (down-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row+i][col+i] == player for i in range(4)):
                    return True
        
        # Diagonal (down-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(self.board[row+i][col-i] == player for i in range(4)):
                    return True
        
        return False


class NeuralNetwork:
    """Simple feedforward neural network"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.01):
        self.lr = learning_rate
        
        # He initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """Forward pass"""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.z2
        return self.a2
    
    def backward(self, x, target, output):
        """Backward pass using MSE loss"""
        m = x.shape[0]
        
        # Output layer gradients
        dz2 = (output - target) / m
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
    
    def predict(self, x):
        return self.forward(x)


class Connect4Agent:
    """RL Agent for Connect 4"""
    def __init__(self, epsilon: float = 0.3, gamma: float = 0.95):
        self.nn = NeuralNetwork(42, 64, 7, learning_rate=0.001)
        self.epsilon = epsilon
        self.gamma = gamma
        
    def flip_state(self, state: np.ndarray) -> np.ndarray:
        """Flip the board perspective (swap 1s and 2s)"""
        flipped = state.copy()
        flipped[state == 1] = 2
        flipped[state == 2] = 1
        return flipped
    
    def get_action(self, state: np.ndarray, valid_moves: List[int], training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        state_flat = state.flatten().reshape(1, -1)
        q_values = self.nn.predict(state_flat)[0]
        
        # Mask invalid moves
        masked_q = np.full(7, -np.inf)
        for move in valid_moves:
            masked_q[move] = q_values[move]
        
        return np.argmax(masked_q)
    
    def train_step(self, state: np.ndarray, action: int, reward: float, 
                   next_state: np.ndarray, done: bool, valid_next_moves: List[int]):
        """Single training step using Q-learning"""
        state_flat = state.flatten().reshape(1, -1)
        next_state_flat = next_state.flatten().reshape(1, -1)
        
        # Current Q-values
        current_q = self.nn.predict(state_flat)
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            next_q = self.nn.predict(next_state_flat)[0]
            masked_next_q = np.full(7, -np.inf)
            for move in valid_next_moves:
                masked_next_q[move] = next_q[move]
            target_q = reward + self.gamma * np.max(masked_next_q)
        
        # Update only the action taken
        target = current_q.copy()
        target[0][action] = target_q
        
        # Backpropagation
        self.nn.backward(state_flat, target, current_q)


def train_selfplay(episodes: int = 2000, verbose: bool = True):
    """
    Train agent through SELF-PLAY
    Both players use the same agent, so it learns from both perspectives
    """
    agent = Connect4Agent(epsilon=0.3)
    env = Connect4()
    
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    for episode in range(episodes):
        state = env.reset(randomize_start=True)  # Randomize who starts!
        done = False
        
        # Store experiences for both players
        p1_history = []  # Player 1's experiences
        p2_history = []  # Player 2's experiences
        
        move_count = 0
        starting_player = env.current_player  # Track who started
        
        while not done:
            current_player = env.current_player
            valid_moves = env.get_valid_moves()
            
            # Get action from agent (same agent for both players!)
            if current_player == 1:
                # Player 1 sees the board as-is
                action = agent.get_action(state, valid_moves)
                next_state, reward, done = env.make_move(action)
                p1_history.append((state.copy(), action, reward, next_state.copy(), done))
            else:
                # Player 2 sees the board FLIPPED (1s become 2s, 2s become 1s)
                # This way the agent always learns from "my pieces vs opponent pieces"
                flipped_state = agent.flip_state(state)
                action = agent.get_action(flipped_state, valid_moves)
                next_state, reward, done = env.make_move(action)
                flipped_next = agent.flip_state(next_state)
                p2_history.append((flipped_state, action, reward, flipped_next, done))
            
            state = next_state
            move_count += 1
            
            # Safety: prevent infinite loops
            if move_count > 100:
                done = True
                reward = 0
        
        # Determine final rewards from each player's perspective
        if reward == 1:  # Last player won
            if env.current_player == 2:  # Player 1 won (switched after move)
                p1_wins += 1
                final_reward_p1 = 1
                final_reward_p2 = -1
            else:  # Player 2 won
                p2_wins += 1
                final_reward_p1 = -1
                final_reward_p2 = 1
        else:  # Draw
            draws += 1
            final_reward_p1 = 0
            final_reward_p2 = 0
        
        # Update final rewards in histories
        if p1_history:
            p1_history[-1] = (p1_history[-1][0], p1_history[-1][1], 
                             final_reward_p1, p1_history[-1][3], True)
        if p2_history:
            p2_history[-1] = (p2_history[-1][0], p2_history[-1][1], 
                             final_reward_p2, p2_history[-1][3], True)
        
        # Propagate rewards backwards (opponent's moves get negative reward)
        for i in range(len(p1_history) - 2, -1, -1):
            state, action, _, next_state, _ = p1_history[i]
            # If opponent won next, this move gets penalized
            if p1_history[i + 1][2] < 0:  # Next move led to opponent win
                p1_history[i] = (state, action, -0.1, next_state, False)
        
        for i in range(len(p2_history) - 2, -1, -1):
            state, action, _, next_state, _ = p2_history[i]
            if p2_history[i + 1][2] < 0:
                p2_history[i] = (state, action, -0.1, next_state, False)
        
        # Train on BOTH players' experiences
        for state, action, reward, next_state, done in p1_history:
            valid_next = [c for c in range(7) if next_state[0][c] == 0]
            agent.train_step(state, action, reward, next_state, done, valid_next)
        
        for state, action, reward, next_state, done in p2_history:
            valid_next = [c for c in range(7) if next_state[0][c] == 0]
            agent.train_step(state, action, reward, next_state, done, valid_next)
        
        # Decay epsilon
        agent.epsilon = max(0.05, agent.epsilon * 0.995)
        
        if verbose and (episode + 1) % 100 == 0:
            total = episode + 1
            print(f"Episode {total}/{episodes}")
            print(f"  P1 Wins: {p1_wins} ({p1_wins/total*100:.1f}%)")
            print(f"  P2 Wins: {p2_wins} ({p2_wins/total*100:.1f}%)")
            print(f"  Draws: {draws} ({draws/total*100:.1f}%)")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg moves per game: {(len(p1_history) + len(p2_history)):.1f}")
            print(f"  (Games randomize starting player for balanced learning)")
            print()
    
    return agent


def play_game(agent: Connect4Agent):
    """Play a game against the trained agent"""
    env = Connect4()
    state = env.reset()
    
    print("\nConnect 4 Game Started!")
    print("You are Player 2 (represented by O)")
    print("Columns are numbered 0-6")
    print_board(state)
    
    while True:
        # Agent's turn (Player 1)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        action = agent.get_action(state, valid_moves, training=False)
        print(f"\nAgent plays column {action}")
        state, reward, done = env.make_move(action)
        print_board(state)
        
        if done:
            if reward == 1:
                print("Agent wins!")
            else:
                print("Draw!")
            break
        
        # Human's turn (Player 2)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        while True:
            try:
                col = int(input(f"\nYour turn! Choose column (0-6): "))
                if col in valid_moves:
                    break
                print(f"Invalid move! Valid columns: {valid_moves}")
            except ValueError:
                print("Please enter a number")
        
        state, reward, done = env.make_move(col)
        print_board(state)
        
        if done:
            if reward == 1:
                print("You win!")
            else:
                print("Draw!")
            break


def print_board(board: np.ndarray):
    """Print the board"""
    print("\n" + "=" * 29)
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print(" . ", end="|")
            elif cell == 1:
                print(" X ", end="|")
            else:
                print(" O ", end="|")
        print()
    print("=" * 29)
    print("  0   1   2   3   4   5   6")


def human_teaching_session(agent: Connect4Agent):
    """Allow human to demonstrate good moves to teach the agent"""
    env = Connect4()
    state = env.reset(randomize_start=False)  # Always start fresh
    
    print("\n" + "="*60)
    print("HUMAN TEACHING MODE")
    print("="*60)
    print("\nYou can demonstrate good moves to help train the agent!")
    print("Commands:")
    print("  - Enter column number (0-6) to make a move")
    print("  - Type 'teach X' to demonstrate that column X is the best move")
    print("  - Type 'reset' to start a new board")
    print("  - Type 'done' to finish teaching")
    print("="*60)
    
    teaching_examples = []
    
    while True:
        print_board(state)
        
        # Show what the agent currently thinks
        valid_moves = env.get_valid_moves()
        state_flat = state.flatten().reshape(1, -1)
        q_values = agent.nn.predict(state_flat)[0]
        
        print("\nAgent's current Q-values for valid moves:")
        for col in valid_moves:
            print(f"  Column {col}: {q_values[col]:.3f}")
        
        agent_choice = valid_moves[np.argmax([q_values[col] for col in valid_moves])]
        print(f"\nAgent would choose: Column {agent_choice}")
        
        # Get human input
        command = input("\nYour command (column/teach X/reset/done): ").strip().lower()
        
        if command == 'done':
            break
        
        if command == 'reset':
            state = env.reset(randomize_start=False)
            print("\nBoard reset!")
            continue
        
        # Teaching command: "teach X"
        if command.startswith('teach'):
            parts = command.split()
            if len(parts) == 2:
                try:
                    best_col = int(parts[1])
                    if best_col in valid_moves:
                        # Save this as a teaching example
                        teaching_examples.append((state.copy(), best_col))
                        print(f"\n✓ Recorded! You taught: Column {best_col} is best here")
                        print(f"  (Agent thought Column {agent_choice} was best)")
                        
                        # Immediately train on this example
                        # Set high Q-value for the taught move
                        state_flat = state.flatten().reshape(1, -1)
                        current_q = agent.nn.predict(state_flat)
                        target = current_q.copy()
                        
                        # Give the taught move a high target Q-value
                        target[0][best_col] = 1.0
                        
                        # Give other moves lower values (especially the wrong choice)
                        for col in valid_moves:
                            if col != best_col:
                                target[0][col] = max(-0.5, current_q[0][col] - 0.3)
                        
                        # Train multiple times on this example for emphasis
                        for _ in range(5):
                            agent.nn.backward(state_flat, target, current_q)
                            current_q = agent.nn.predict(state_flat)
                        
                        print("  Agent has been updated with this knowledge!")
                        continue
                    else:
                        print(f"Column {best_col} is not valid! Valid: {valid_moves}")
                        continue
                except ValueError:
                    print("Invalid format. Use: teach 3")
                    continue
            else:
                print("Invalid format. Use: teach 3")
                continue
        
        # Regular move
        try:
            col = int(command)
            if col not in valid_moves:
                print(f"Invalid move! Valid columns: {valid_moves}")
                continue
            
            # Make the move
            state, reward, done = env.make_move(col)
            
            if done:
                print_board(state)
                if reward == 1:
                    print(f"\nPlayer {3 - env.current_player} wins!")
                else:
                    print("\nDraw!")
                state = env.reset(randomize_start=False)
                print("\nStarting new board...")
            
        except ValueError:
            print("Invalid command. Try: 0-6, teach X, reset, or done")
    
    print(f"\n✓ Teaching session complete! You provided {len(teaching_examples)} examples.")
    return teaching_examples


def play_with_hints(agent: Connect4Agent):
    """Play against agent with option to see/teach during game"""
    env = Connect4()
    state = env.reset()
    
    print("\n" + "="*60)
    print("PLAY WITH TEACHING MODE")
    print("="*60)
    print("You are Player 2 (O)")
    print("Commands:")
    print("  - Enter column (0-6) to play")
    print("  - Type 'hint' to see agent's thinking")
    print("  - Type 'teach X' when agent plays badly to correct it")
    print("="*60)
    
    print_board(state)
    
    while True:
        # Agent's turn (Player 1)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        action = agent.get_action(state, valid_moves, training=False)
        print(f"\nAgent plays column {action}")
        
        agent_state = state.copy()
        state, reward, done = env.make_move(action)
        print_board(state)
        
        if done:
            if reward == 1:
                print("Agent wins!")
            else:
                print("Draw!")
            break
        
        # Ask if human wants to teach about that move
        teach_input = input("\nWas that move bad? Type 'teach X' to show better move, or press ENTER: ").strip().lower()
        if teach_input.startswith('teach'):
            parts = teach_input.split()
            if len(parts) == 2:
                try:
                    better_col = int(parts[1])
                    # Undo the agent's move
                    env.board = agent_state
                    env.current_player = 1
                    state = agent_state
                    
                    valid_undo = env.get_valid_moves()
                    if better_col in valid_undo:
                        # Train agent that better_col was better
                        state_flat = state.flatten().reshape(1, -1)
                        current_q = agent.nn.predict(state_flat)
                        target = current_q.copy()
                        target[0][better_col] = 1.0
                        target[0][action] = -0.5  # Penalize the bad move
                        
                        for _ in range(10):
                            agent.nn.backward(state_flat, target, current_q)
                            current_q = agent.nn.predict(state_flat)
                        
                        print(f"✓ Agent learned! Column {better_col} is better than {action}")
                        
                        # Make the better move
                        state, reward, done = env.make_move(better_col)
                        print(f"\nReplaying with column {better_col}:")
                        print_board(state)
                        
                        if done:
                            if reward == 1:
                                print("Agent wins!")
                            else:
                                print("Draw!")
                            break
                except ValueError:
                    pass
        
        # Human's turn (Player 2)
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("Draw!")
            break
        
        while True:
            move_input = input(f"\nYour turn! (0-6, or 'hint'): ").strip().lower()
            
            if move_input == 'hint':
                # Show agent's evaluation of the position (from Player 2's perspective)
                flipped = agent.flip_state(state)
                state_flat = flipped.flatten().reshape(1, -1)
                q_values = agent.nn.predict(state_flat)[0]
                
                print("\nAgent's Q-values for YOUR position:")
                for col in valid_moves:
                    print(f"  Column {col}: {q_values[col]:.3f}")
                best = valid_moves[np.argmax([q_values[col] for col in valid_moves])]
                print(f"Agent suggests: Column {best}")
                continue
            
            try:
                col = int(move_input)
                if col in valid_moves:
                    break
                print(f"Invalid! Valid: {valid_moves}")
            except ValueError:
                print("Enter a number 0-6 or 'hint'")
        
        state, reward, done = env.make_move(col)
        print_board(state)
        
        if done:
            if reward == 1:
                print("You win!")
            else:
                print("Draw!")
            break


def get_training_parameters():
    """Interactive parameter configuration"""
    print("="*60)
    print("CONNECT 4 NEURAL NETWORK TRAINING CONFIGURATOR")
    print("="*60)
    print("\nPress ENTER to use default values shown in [brackets]\n")
    
    # Episodes
    while True:
        try:
            episodes_input = input("Number of training episodes [2000]: ").strip()
            episodes = int(episodes_input) if episodes_input else 2000
            if episodes > 0:
                break
            print("Must be positive!")
        except ValueError:
            print("Please enter a valid number")
    
    # Learning rate
    while True:
        try:
            lr_input = input("Learning rate (0.0001 to 0.1) [0.001]: ").strip()
            learning_rate = float(lr_input) if lr_input else 0.001
            if 0.0001 <= learning_rate <= 0.1:
                break
            print("Must be between 0.0001 and 0.1")
        except ValueError:
            print("Please enter a valid number")
    
    # Epsilon (exploration rate)
    while True:
        try:
            eps_input = input("Starting epsilon/exploration (0.0 to 1.0) [0.3]: ").strip()
            epsilon = float(eps_input) if eps_input else 0.3
            if 0.0 <= epsilon <= 1.0:
                break
            print("Must be between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    # Gamma (discount factor)
    while True:
        try:
            gamma_input = input("Gamma/discount factor (0.0 to 1.0) [0.95]: ").strip()
            gamma = float(gamma_input) if gamma_input else 0.95
            if 0.0 <= gamma <= 1.0:
                break
            print("Must be between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")
    
    # Hidden layer size
    while True:
        try:
            hidden_input = input("Hidden layer neurons (16 to 256) [64]: ").strip()
            hidden_size = int(hidden_input) if hidden_input else 64
            if 16 <= hidden_size <= 256:
                break
            print("Must be between 16 and 256")
        except ValueError:
            print("Please enter a valid number")
    
    # Epsilon decay
    while True:
        try:
            decay_input = input("Epsilon decay rate (0.9 to 0.999) [0.995]: ").strip()
            epsilon_decay = float(decay_input) if decay_input else 0.995
            if 0.9 <= epsilon_decay <= 0.999:
                break
            print("Must be between 0.9 and 0.999")
        except ValueError:
            print("Please enter a valid number")
    
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY:")
    print("="*60)
    print(f"Episodes:              {episodes}")
    print(f"Learning Rate:         {learning_rate}")
    print(f"Starting Epsilon:      {epsilon}")
    print(f"Gamma (Discount):      {gamma}")
    print(f"Hidden Layer Size:     {hidden_size}")
    print(f"Epsilon Decay:         {epsilon_decay}")
    print("="*60)
    
    confirm = input("\nStart training with these parameters? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled.")
        return None
    
    return {
        'episodes': episodes,
        'learning_rate': learning_rate,
        'epsilon': epsilon,
        'gamma': gamma,
        'hidden_size': hidden_size,
        'epsilon_decay': epsilon_decay
    }


if __name__ == "__main__":
    print("="*60)
    print("SELF-PLAY TRAINING")
    print("="*60)
    print("The agent plays BOTH sides and learns from both perspectives!")
    print("This dramatically improves learning speed and final strength.\n")
    
    # Get parameters from user
    params = get_training_parameters()
    
    if params is None:
        exit()
    
    # Create agent with custom parameters
    agent = Connect4Agent(epsilon=params['epsilon'], gamma=params['gamma'])
    agent.nn = NeuralNetwork(42, params['hidden_size'], 7, 
                            learning_rate=params['learning_rate'])
    
    # Store epsilon decay for use in training
    epsilon_decay = params['epsilon_decay']
    
    # Modified training loop with custom epsilon decay
    env = Connect4()
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    print("\n" + "="*60)
    print("TRAINING IN PROGRESS...")
    print("="*60 + "\n")
    
    for episode in range(params['episodes']):
        state = env.reset(randomize_start=True)
        done = False
        
        p1_history = []
        p2_history = []
        move_count = 0
        starting_player = env.current_player
        
        while not done:
            current_player = env.current_player
            valid_moves = env.get_valid_moves()
            
            if current_player == 1:
                action = agent.get_action(state, valid_moves)
                next_state, reward, done = env.make_move(action)
                p1_history.append((state.copy(), action, reward, next_state.copy(), done))
            else:
                flipped_state = agent.flip_state(state)
                action = agent.get_action(flipped_state, valid_moves)
                next_state, reward, done = env.make_move(action)
                flipped_next = agent.flip_state(next_state)
                p2_history.append((flipped_state, action, reward, flipped_next, done))
            
            state = next_state
            move_count += 1
            
            if move_count > 100:
                done = True
                reward = 0
        
        # Determine final rewards
        if reward == 1:
            if env.current_player == 2:
                p1_wins += 1
                final_reward_p1 = 1
                final_reward_p2 = -1
            else:
                p2_wins += 1
                final_reward_p1 = -1
                final_reward_p2 = 1
        else:
            draws += 1
            final_reward_p1 = 0
            final_reward_p2 = 0
        
        # Update final rewards
        if p1_history:
            p1_history[-1] = (p1_history[-1][0], p1_history[-1][1], 
                             final_reward_p1, p1_history[-1][3], True)
        if p2_history:
            p2_history[-1] = (p2_history[-1][0], p2_history[-1][1], 
                             final_reward_p2, p2_history[-1][3], True)
        
        # Propagate rewards
        for i in range(len(p1_history) - 2, -1, -1):
            state_h, action_h, _, next_state_h, _ = p1_history[i]
            if p1_history[i + 1][2] < 0:
                p1_history[i] = (state_h, action_h, -0.1, next_state_h, False)
        
        for i in range(len(p2_history) - 2, -1, -1):
            state_h, action_h, _, next_state_h, _ = p2_history[i]
            if p2_history[i + 1][2] < 0:
                p2_history[i] = (state_h, action_h, -0.1, next_state_h, False)
        
        # Train on both players
        for state_t, action_t, reward_t, next_state_t, done_t in p1_history:
            valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
            agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
        
        for state_t, action_t, reward_t, next_state_t, done_t in p2_history:
            valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
            agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
        
        # Decay epsilon with custom rate
        agent.epsilon = max(0.05, agent.epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            total = episode + 1
            print(f"Episode {total}/{params['episodes']}")
            print(f"  P1 Wins: {p1_wins} ({p1_wins/total*100:.1f}%)")
            print(f"  P2 Wins: {p2_wins} ({p2_wins/total*100:.1f}%)")
            print(f"  Draws: {draws} ({draws/total*100:.1f}%)")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print()
    
    trained_agent = agent
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Statistics:")
    print(f"  Total Games: {params['episodes']}")
    print(f"  P1 Win Rate: {p1_wins/params['episodes']*100:.1f}%")
    print(f"  P2 Win Rate: {p2_wins/params['episodes']*100:.1f}%")
    print(f"  Draw Rate: {draws/params['episodes']*100:.1f}%")
    print(f"  Final Epsilon: {agent.epsilon:.3f}")
    print("="*60)
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Play against the agent")
        print("2. Play with teaching mode (correct agent during game)")
        print("3. Teaching session (show agent good positions)")
        print("4. Train more episodes")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            play_game(trained_agent)
        elif choice == '2':
            play_with_hints(trained_agent)
        elif choice == '3':
            human_teaching_session(trained_agent)
        elif choice == '4':
            # Additional training
            while True:
                try:
                    extra_episodes = int(input("How many more episodes? "))
                    if extra_episodes > 0:
                        break
                except ValueError:
                    pass
            
            print(f"\nTraining {extra_episodes} more episodes...")
            
            # Continue training with same agent
            for episode in range(extra_episodes):
                state = env.reset(randomize_start=True)
                done = False
                
                p1_history = []
                p2_history = []
                move_count = 0
                
                while not done:
                    current_player = env.current_player
                    valid_moves = env.get_valid_moves()
                    
                    if current_player == 1:
                        action = trained_agent.get_action(state, valid_moves)
                        next_state, reward, done = env.make_move(action)
                        p1_history.append((state.copy(), action, reward, next_state.copy(), done))
                    else:
                        flipped_state = trained_agent.flip_state(state)
                        action = trained_agent.get_action(flipped_state, valid_moves)
                        next_state, reward, done = env.make_move(action)
                        flipped_next = trained_agent.flip_state(next_state)
                        p2_history.append((flipped_state, action, reward, flipped_next, done))
                    
                    state = next_state
                    move_count += 1
                    
                    if move_count > 100:
                        done = True
                        reward = 0
                
                if reward == 1:
                    if env.current_player == 2:
                        final_reward_p1 = 1
                        final_reward_p2 = -1
                    else:
                        final_reward_p1 = -1
                        final_reward_p2 = 1
                else:
                    final_reward_p1 = 0
                    final_reward_p2 = 0
                
                if p1_history:
                    p1_history[-1] = (p1_history[-1][0], p1_history[-1][1], 
                                     final_reward_p1, p1_history[-1][3], True)
                if p2_history:
                    p2_history[-1] = (p2_history[-1][0], p2_history[-1][1], 
                                     final_reward_p2, p2_history[-1][3], True)
                
                for i in range(len(p1_history) - 2, -1, -1):
                    state_h, action_h, _, next_state_h, _ = p1_history[i]
                    if p1_history[i + 1][2] < 0:
                        p1_history[i] = (state_h, action_h, -0.1, next_state_h, False)
                
                for i in range(len(p2_history) - 2, -1, -1):
                    state_h, action_h, _, next_state_h, _ = p2_history[i]
                    if p2_history[i + 1][2] < 0:
                        p2_history[i] = (state_h, action_h, -0.1, next_state_h, False)
                
                for state_t, action_t, reward_t, next_state_t, done_t in p1_history:
                    valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
                    trained_agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
                
                for state_t, action_t, reward_t, next_state_t, done_t in p2_history:
                    valid_next = [c for c in range(7) if next_state_t[0][c] == 0]
                    trained_agent.train_step(state_t, action_t, reward_t, next_state_t, done_t, valid_next)
                
                trained_agent.epsilon = max(0.05, trained_agent.epsilon * epsilon_decay)
            
            print(f"✓ Trained {extra_episodes} more episodes!")
            
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice!")
