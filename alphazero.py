import torch
import torch.nn as nn
from utils import board_to_tensor


class ResnetBlock(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, padding=1
        )  # [batch_size, 12, 8, 8] -> [batch_size, 256, 8, 8]
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(
            output_dim, output_dim, kernel_size=3, padding=1
        )  # [batch_size, 256, 8, 8] -> [batch_size, 256, 8, 8]
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self, input_dim=12, num_blocks=10, num_actions=4672):
        super(AlphaZeroNet, self).__init__()
        self.conv = nn.Conv2d(
            input_dim, 256, kernel_size=3, padding=1
        )  # [batch_size, 12, 8, 8] -> [batch_size, 256, 8, 8]
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.resnet_blocks = nn.ModuleList(
            [ResnetBlock(256) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(
            256, 2, kernel_size=1
        )  # [batch_size, 256, 8, 8] -> [batch_size, 2, 8, 8]
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * 8 * 8, num_actions
        )  # [batch_size, 128] -> [batch_size, num_actions]

        # Value head
        self.value_conv = nn.Conv2d(
            256, 1, kernel_size=1
        )  # [batch_size, 256, 8, 8] -> [batch_size, 1, 8, 8]
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(
            1 * 8 * 8, 256
        )  # [batch_size, 64] -> [batch_size, 256]
        self.value_fc2 = nn.Linear(256, 1)  # [batch_size, 256] -> [batch_size, 1]
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        for block in self.resnet_blocks:
            x = block(x)

        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.relu(policy)
        policy = policy.view(
            policy.size(0), -1
        )  # [batch_size, 2, 8, 8] -> [batch_size, 128]
        policy = self.policy_fc(policy)

        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.relu(value)
        value = value.view(
            value.size(0), -1
        )  # [batch_size, 1, 8, 8] -> [batch_size, 64]
        value = self.value_fc1(value)
        value = self.relu(value)
        value = self.value_fc2(value)
        value = self.tanh(value)

        return policy, value


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior_probability = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.total_value / (child.visit_count + 1e-8))
            + c_param
            * child.prior_probability
            * ((self.visit_count**0.5) / (1 + child.visit_count))
            for child in self.children.values()
        ]
        return list(self.children.values())[choices_weights.index(max(choices_weights))]

    def expand(self, action, next_state, prior_probability):
        if action not in self.children:
            child_node = MCTSNode(next_state, parent=self)
            child_node.prior_probability = prior_probability
            self.children[action] = child_node

    def update(self, value):
        self.visit_count += 1
        self.total_value += value


class MCTS:
    def __init__(self, neural_net, num_simulations=800, device="cpu"):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.device = device

    def search(self, initial_state):
        root = MCTSNode(initial_state)
        state_tensor = board_to_tensor(initial_state).unsqueeze(0).to(self.device)
        policy, _ = self.neural_net(state_tensor)
        policy = torch.softmax(policy, dim=1).detach().cpu().numpy()[0]

        legal_moves = list(initial_state.legal_moves)
        for move in legal_moves:
            next_state = initial_state.copy()
            next_state.push(move)
            # Use a simple prior for now (uniform distribution over legal moves)
            prior = 1.0 / len(legal_moves)
            root.expand(move, next_state, prior)

        for _ in range(self.num_simulations):
            node = root
            state = initial_state
            path = []

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                path.append(node)

            # Get the state corresponding to this node
            if path:
                state = path[-1].state

            # Expansion
            if not node.is_fully_expanded() and not state.is_game_over():
                legal_moves = list(state.legal_moves)
                for move in legal_moves:
                    if move not in node.children:
                        next_state = state.copy()
                        next_state.push(move)
                        state_tensor = (
                            board_to_tensor(state).unsqueeze(0).to(self.device)
                        )
                        policy, value = self.neural_net(state_tensor)
                        policy = torch.softmax(policy, dim=1).detach().cpu().numpy()[0]
                        prior = 1.0 / len(legal_moves)  # Uniform prior for now
                        node.expand(move, next_state, prior)
                        node = node.children[move]
                        break

            # Simulation (evaluation with neural network)
            node_state_tensor = board_to_tensor(node.state).unsqueeze(0).to(self.device)
            _, value = self.neural_net(node_state_tensor)
            value = value.item()

            # Backpropagation
            while node is not None:
                node.update(value)
                node = node.parent
                value = -value  # Flip value for opponent

        # Return action with most visits
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]


class SelfPlay:
    def __init__(
        self, neural_net, env, replay_buffer, mcts_simulations=800, device="cpu"
    ):
        self.neural_net = neural_net
        self.env = env
        self.replay_buffer = replay_buffer
        self.device = device
        self.mcts = MCTS(neural_net, num_simulations=mcts_simulations, device=device)

    def generate_game(self):
        state = self.env.reset()
        done = False
        game_data = []

        while not done:
            action = self.mcts.search(state)
            state_tensor = board_to_tensor(state).unsqueeze(0).to(self.device)
            policy, value = self.neural_net(state_tensor)
            policy = torch.softmax(policy, dim=1).detach().cpu().numpy()[0]
            game_data.append(
                (board_to_tensor(state), policy, None)
            )  # Value to be filled later

            state, reward, done, _ = self.env.step(action)

        # Assign values to each state in the game data
        for i in range(len(game_data)):
            game_data[i] = (
                game_data[i][0],
                game_data[i][1],
                reward if (i % 2 == 0) else -reward,
            )
            self.replay_buffer.add(*game_data[i])  # Add to replay buffer

        return game_data
