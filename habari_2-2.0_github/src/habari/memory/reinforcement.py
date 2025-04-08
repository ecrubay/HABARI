
# reinforcement.py

class FeedbackReinforcement:
    def __init__(self):
        self.history = []

    def record_interaction(self, result):
        self.history.append(result)

    def calculate_success_rate(self):
        success = sum(1 for x in self.history if x.get("success", False))
        return success / len(self.history) if self.history else 0.0
