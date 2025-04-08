
# insights.py

def generate_report(interactions):
    total = len(interactions)
    success = sum(1 for i in interactions if i['success'])
    return {
        "total": total,
        "success": success,
        "rate": round(success / total, 2) if total > 0 else 0
    }
