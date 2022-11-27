import torch

class SimpleDealer:
    def __init__(self, table, brain):
        self.table = table
        self.brain = brain

    def game(self, n_games, n_players, relocation_freq = 180, training_freq = 45):
        self.n_players = n_players
        for game in range(n_games):
            if game % relocation_freq == 0:
                self.brain.sit(n_players)

            self.game__()

            if game % training_freq == training_freq - 1:
                self.brain.optimize()            

    def game__(self):
        end = False
        self.table.start_table()
        table_state = self.table.get_state()
        action = torch.zeros((self.table.bins + 4,))
        env_state = self.prepare_state({"table_state": [], "now": 0}, table_state, action)
        actions = [[]] * self.n_players
        while not end:
            action = self.brain.step(self.table.active_player, env_state)
            actions[self.table.active_player].append(action)
            env_state = self.prepare_state(env_state, table_state, action["action"])
            end, table_state = self.table.step(action["action"])
            env_state["now"] = table_state
            env_state["active_positions"] = table_state["active_positions"]

        reward = self.table.get_reward()
        self.brain.backward(actions, reward)
        self.brain.rotate()
        for i in range(self.n_players):
            if self.table.credits[i] == 0:
                self.table.reset(i)

    def prepare_state(self, env_state, table_state, action):
        for_env = {
            "pos": table_state["pos"],
            "pot": table_state["pot"],
            "action": action,
            "table": table_state["table"]
        }
        env_state["table_state"].append(for_env)
        return env_state