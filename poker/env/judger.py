import numpy as np

class Judger:
    def __init__(self):
        pass

    def get_reward(self, deck, players_state, bets):
        active_players = players_state >= 0
        if active_players.sum() == 1:
            bets[np.argmax(active_players)] = np.sum(bets)
        else:
            active_players_numbers = np.arange(len(players_state))[active_players]
            bets[active_players_numbers] = self.share_out(deck, active_players_numbers, bets)
        
        return bets

    def share_out(self, deck, active_players_numbers, bets):
        n_players = len(bets)
        rewards = np.zeros((n_players,))
        hands = np.array([deck[:5] + deck[5 + i * 2 : 7 + i * 2] for i in active_players_numbers])
        powers = self.eval_hands(active_players_numbers, n_players, hands)
        while np.max(powers) > 0:
            potential_winners = powers.sum(axis = 1) == np.max(powers.sum(axis = 1))
            winners = np.arange(n_players)[potential_winners]
            while np.max(winners) > -1:
                pie = np.min(bets[winners])
                pay_now = winners[bets[winners] >= pie]
                rewards[pay_now] += pie
                bets[pay_now] -= pie
                for i in range(n_players):
                    if i in pay_now:
                        continue
                    pie_ = min(bets[i], pie)
                    rewards[i] -= pie_
                    bets[i] -= pie_
                    rewards[pay_now] += pie_ / len(pay_now)
                winners[bets[winners] == pie] = -1
            powers[potential_winners] -= 1
        
        return rewards

    def eval_hands(self, active_players_numbers, n_players, hands):
        powers = np.zeros((n_players, n_players))
        for i in active_players_numbers:
            for j in active_players_numbers:
                if (i >= j): continue
                powers[i][j], powers[j][i] = self.compare_hands(hands[i], hands[j])

        return powers

    def compare_hands(self, hand_1, hand_2):
        hand_1, hand_2 = np.sort(hand_1), np.sort(hand_2)
        (p_1, bord_1), (p_2, bord_2) = self.compute_power(hand_1), self.compute_power(hand_2)
        if p_1 > p_2: return 1, 0
        elif p_2 > p_1: return 0, 1
        else:
            temp_1, temp_2 = bord_1 > bord_2, bord_1 == bord_2
            if temp_1: return 1, 0
            elif temp_2: return 1, 1
            else: return 0, 1

    def compute_power(self, hand):
        rank, suit = hand // 4, hand % 4
        flush, straight, fourakind, threeakind, twoakind, twopairs, count = False, False, False, False, False, False, 1
        for i in range(4):
            if (suit == i).sum() >= 5: flush = True
        for i in range(6):
            if rank[i] + 1 == rank[i + 1]: 
                count += 1
                if count == 5: straight = True
            else: count = 1
        for r in rank:
            if (rank == r).sum() == 4: fourakind = True
            if (rank == r).sum() == 3: threeakind = True
            if (rank == r).sum() == 2 and twoakind: twopairs = True
            if (rank == r).sum() == 2: twoakind = True

        if flush and straight: return 8, self.get_bord(8, rank, suit)
        if fourakind: return 7, self.get_bord(7, rank, suit)
        if threeakind and twopairs: return 6, self.get_bord(6, rank, suit)
        if flush: return 5, self.get_bord(5, rank, suit)
        if straight: return 4, self.get_bord(4, rank, suit)
        if threeakind: return 3, self.get_bord(3, rank, suit)
        if twopairs: return 2, self.get_bord(2, rank, suit)
        if twoakind: return 1, self.get_bord(1, rank, suit)

        return 0, np.flip(hand[2:])

    def get_bord(self, power, rank, suit):
        bord = []
        if power == 8: 
            bord = [rank[6] * 4 + suit[6]]
            for i in range(6, 0, -1):
                if rank[i] == rank[i - 1] + 1 and suit[i] == suit[i - 1]: bord.append(rank[i - 1] * 4 + suit[i - 1])
                else: bord = [rank[i - 1] * 4 + suit[i - 1]]
            bord = bord[- 5:]

        if power == 7:
            for i in range(3):
                if (rank == rank[i]).sum() == 4: 
                    bord = list(rank[i : i + 4] * 4 + suit[i : i + 4]) + [rank[-1] * 4 + suit[-1]]
                    break
            if bord == []: bord = [rank[2] * 4 + suit[2]] + list(rank[3:] * 4 + suit[3:])

        if power == 6:
            three, two = False, False
            for i in range(6, -1, -1):
                if (rank == rank[i]).sum() == 3 and not three: 
                    bord += list(rank[rank == rank[i]] * 4 + suit[rank == rank[i]])
                    three = True
                if (rank == rank[i]).sum() == 2 and not two: 
                    bord += list(rank[rank == rank[i]] * 4 + suit[rank == rank[i]])
                    two = True

        if power == 5:
            for i in range(4):
                if (suit == i).sum() == 5:
                    bord = rank[suit == i] * 4 + suit[suit == i]
            bord = bord[- 5:]

        if power == 4:
            bord = [rank[6] * 4 + suit[6]]
            for i in range(6, 0, -1):
                if rank[i] == rank[i - 1] + 1: bord.append(rank[i - 1] * 4 + suit[i - 1])
                else: bord = [rank[i - 1] * 4 + suit[i - 1]]
            bord = bord[- 5:]

        if power == 3:
            for i in range(7):
                if (rank == rank[i]).sum() == 3: 
                    bord = list(rank[rank == rank[i]] * 4 + suit[rank == rank[i]]) + list(rank[~(rank == rank[i])] * 4 + suit[~(rank == rank[i])])[-2:]
                    break

        if power == 2:
            for i in range(7):
                if (rank == rank[i]).sum() == 2 and np.all(rank[rank == rank[i]] > -1): 
                    bord += list(rank[rank == rank[i]] * 4 + suit[rank == rank[i]])
                    rank[rank == rank[i]] -= 10000
            for i in range(6, -1, -1):
                if rank[i] * 4 + suit[i] > -1 and rank[i] * 4 + suit[i] not in bord:
                    bord.append(rank[i] * 4 + suit[i])
                    break

        if power == 1:
            for i in range(7):
                if (rank == rank[i]).sum() == 2: 
                    bord = list(rank[rank == rank[i]] * 4 + suit[rank == rank[i]]) + list(rank[~(rank == rank[i])] * 4 + suit[~(rank == rank[i])])[-3:]
                    break

        return np.flip(np.sort(bord))
            
                


        
        






