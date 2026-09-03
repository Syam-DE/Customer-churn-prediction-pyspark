"""
Generates a synthetic customer dataset for a subscription/meal-kit style
business (tenure, engagement, delivery, support, pricing signals) with a
realistic churn label baked in via a weighted logistic function + noise.

Run: python src/generate_data.py
Output: data/customers.csv
"""

import csv
import random
import math

random.seed(42)

N_CUSTOMERS = 20000

PLAN_TYPES = ["2-person-3meal", "2-person-4meal", "4-person-2meal", "4-person-4meal"]
ACQUISITION_CHANNELS = ["paid_social", "referral", "organic_search", "affiliate", "email"]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def generate_customer(customer_id):
    tenure_months = max(1, int(random.gauss(10, 7)))
    plan_type = random.choice(PLAN_TYPES)
    acquisition_channel = random.choices(
        ACQUISITION_CHANNELS, weights=[0.35, 0.15, 0.25, 0.15, 0.10]
    )[0]

    weekly_deliveries_last_8w = max(0, min(8, int(random.gauss(5.5, 2.2))))
    skipped_weeks_last_8w = max(0, min(8 - weekly_deliveries_last_8w, int(random.gauss(1.5, 1.5))))

    avg_box_price = round(random.gauss(11.5, 2.0), 2)
    discount_pct_last_order = round(max(0, random.gauss(8, 10)), 1)

    support_tickets_last_90d = max(0, int(random.gauss(0.8, 1.3)))
    avg_ticket_resolution_hrs = round(max(0.5, random.gauss(14, 10)), 1)
    late_deliveries_last_90d = max(0, int(random.gauss(0.6, 1.0)))

    app_sessions_last_30d = max(0, int(random.gauss(6, 5)))
    recipe_swaps_last_30d = max(0, int(random.gauss(1.2, 1.5)))
    days_since_last_login = max(0, int(random.gauss(6, 8)))

    referred_friends_total = max(0, int(random.gauss(0.3, 0.8)))
    loyalty_tier = random.choices(["none", "silver", "gold"], weights=[0.6, 0.3, 0.1])[0]

    # weighted latent churn score -> higher = more likely to churn
    z = 0.0
    z += -0.06 * tenure_months
    z += -0.35 * weekly_deliveries_last_8w
    z += 0.30 * skipped_weeks_last_8w
    z += 0.28 * support_tickets_last_90d
    z += 0.015 * avg_ticket_resolution_hrs
    z += 0.35 * late_deliveries_last_90d
    z += 0.05 * days_since_last_login
    z += -0.12 * app_sessions_last_30d
    z += -0.4 * referred_friends_total
    z += -0.5 if loyalty_tier == "gold" else (-0.2 if loyalty_tier == "silver" else 0.0)
    z += -0.03 * discount_pct_last_order
    z += random.gauss(0, 1.4)  # noise

    churn_prob = sigmoid(z + 1.0)
    churned = 1 if random.random() < churn_prob else 0

    return {
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "plan_type": plan_type,
        "acquisition_channel": acquisition_channel,
        "weekly_deliveries_last_8w": weekly_deliveries_last_8w,
        "skipped_weeks_last_8w": skipped_weeks_last_8w,
        "avg_box_price": avg_box_price,
        "discount_pct_last_order": discount_pct_last_order,
        "support_tickets_last_90d": support_tickets_last_90d,
        "avg_ticket_resolution_hrs": avg_ticket_resolution_hrs,
        "late_deliveries_last_90d": late_deliveries_last_90d,
        "app_sessions_last_30d": app_sessions_last_30d,
        "recipe_swaps_last_30d": recipe_swaps_last_30d,
        "days_since_last_login": days_since_last_login,
        "referred_friends_total": referred_friends_total,
        "loyalty_tier": loyalty_tier,
        "churned": churned,
    }


def main():
    rows = [generate_customer(i) for i in range(1, N_CUSTOMERS + 1)]
    fieldnames = list(rows[0].keys())

    with open("data/customers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    churn_rate = sum(r["churned"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} rows to data/customers.csv")
    print(f"Overall churn rate: {churn_rate:.2%}")


if __name__ == "__main__":
    main()
