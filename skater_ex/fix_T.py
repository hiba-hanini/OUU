import math
from skater import SkaterMDP


# ============================================================
# Paramètres
# ============================================================

n_values = [7, 10, 15]
obstacle_probs = [0.1, 0.2, 0.3]
p_values = [0.5, 0.7, 0.8, 0.9, 1.0]

initial_seed = 42

# Seuil pour considérer qu'on a atteint le plateau
THRESHOLD = 0.99


# ============================================================
# Chercher une grille valide
# ============================================================

def find_valid_seed(n, obstacle_prob, initial_seed=42):

    seed = initial_seed

    while True:

        skater = SkaterMDP(
            n=n,
            p=0.8,
            obstacle_prob=obstacle_prob,
            seed=seed
        )

        L = skater.shortest_path_length()

        if L is not None:
            return seed, L

        seed += 1


# ============================================================
# Calcul de T*
# ============================================================

results = []


for n in n_values:

    for obstacle_prob in obstacle_probs:

        seed, L = find_valid_seed(
            n,
            obstacle_prob,
            initial_seed
        )

        print(
            f"\nn={n}, obstacle={obstacle_prob}, "
            f"seed={seed}, L={L}"
        )

        for p in p_values:

            skater = SkaterMDP(
                n=n,
                p=p,
                obstacle_prob=obstacle_prob,
                seed=seed
            )

            # ------------------------------------------------
            # Calcul de V(T) pour tous les T
            # ------------------------------------------------

            values_T = {}

            for T in range(L, n**2 + 1):

                values, policies = skater.solve(T)

                V0 = values[0][(skater.start, None)]

                values_T[T] = V0

            # ------------------------------------------------
            # Plateau
            # ------------------------------------------------

            V_max = values_T[n**2]

            target = THRESHOLD * V_max

            # ------------------------------------------------
            # Plus petit T atteignant 99% du plateau
            # ------------------------------------------------

            T_star = None

            for T, V in values_T.items():

                if V >= target:

                    T_star = T
                    break

            # ------------------------------------------------
            # Ratios utiles
            # ------------------------------------------------

            T_over_L = T_star / L

            T_over_Lp = T_star / (L / p)

            results.append({
                "n": n,
                "obstacle_prob": obstacle_prob,
                "p": p,
                "seed": seed,
                "L": L,
                "V_max": V_max,
                "T_star": T_star,
                "T*/L": T_over_L,
                "T*/(L/p)": T_over_Lp
            })

            print(
                f"p={p:.1f} | "
                f"L={L} | "
                f"Vmax={V_max:.3f} | "
                f"T*={T_star} | "
                f"T*/L={T_over_L:.2f} | "
                f"T*/(L/p)={T_over_Lp:.2f}"
            )


# ============================================================
# Affichage du tableau
# ============================================================

print("\n")
print("=" * 100)
print("RESULTATS")
print("=" * 100)

print(
    f"{'n':>3} "
    f"{'obs':>5} "
    f"{'p':>4} "
    f"{'L':>4} "
    f"{'Vmax':>7} "
    f"{'T*':>5} "
    f"{'T*/L':>7} "
    f"{'T*/(L/p)':>10}"
)

print("-" * 100)

for r in results:

    print(
        f"{r['n']:>3} "
        f"{r['obstacle_prob']:>5.1f} "
        f"{r['p']:>4.1f} "
        f"{r['L']:>4} "
        f"{r['V_max']:>7.3f} "
        f"{r['T_star']:>5} "
        f"{r['T*/L']:>7.2f} "
        f"{r['T*/(L/p)']:>10.2f}"
    )