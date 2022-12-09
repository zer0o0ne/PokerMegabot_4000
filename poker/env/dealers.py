import torch

class SimpleDealer:
    def __init__(self, table, brain):
        self.table = table
        self.brain = brain

    def game(self, n_games, n_players, relocation_freq = 180, training_freq = 45, device = "cpu"):
        self.n_players = n_players
        self.device = device
        self.brain.sit(n_players)
        self.brain.set_device(device)
        self.init_history__()

        for game in range(n_games):
            if game % relocation_freq == relocation_freq - 1:
                self.brain.sit(n_players)
                self.init_history__()

            self.game__()

            if game % training_freq == training_freq - 1:
                self.brain.optimize()    

            print("game ", game, " is finished!")        

    def game__(self):
        end = False
        self.table.start_table()
        table_state = self.table.get_state()
        action = torch.zeros((self.table.bins + 4,)).to(self.device)
        env_state = self.prepare_state({"table_state": [], "now": 0}, table_state, action)
        env_state["now"] = table_state
        env_state["active_positions"] = table_state["active_positions"]
        actions = [[] for _ in range(self.n_players)] 
        step = 1
        while not end:
            action = self.brain.step(self.table.active_player, env_state)
            actions[self.table.active_player].append(action)
            env_state = self.prepare_state(env_state, table_state, action["action"])
            end, table_state = self.table.step(action["action"])
            env_state["now"] = table_state
            env_state["active_positions"] = table_state["active_positions"]
            step += 1
        reward = self.table.get_reward()
        self.brain.save_loss(actions, reward)
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

    def init_history__(self):
        self.table.start_table()
        action = torch.zeros((self.table.bins + 4,)).to(self.device)
        table_state = self.table.get_state()
        env_state = self.prepare_state({"table_state": [], "now": 0}, table_state, action)
        env_state["now"] = table_state
        env_state["active_positions"] = table_state["active_positions"]
        self.brain.init_history__(env_state, action, self.n_players)