from config import STARTING_BALANCE, TRADE_SIZE

class RiskManager:
    def __init__(self, balance=STARTING_BALANCE):
        self.balance = balance
        self.trade_size = TRADE_SIZE

    def can_trade(self):
        return self.balance >= self.trade_size

    def apply_result(self, win: bool, payout=0.8):
        if win:
            self.balance += self.trade_size * payout
        else:
            self.balance -= self.trade_size
        return self.balance
