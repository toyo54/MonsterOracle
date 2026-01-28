# ==============================================================================
#  D&D 5e CR PREDICTOR - FINAL DEMO (Strict Blindfold Model)
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

# Configurazione pulizia output
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

print(" D&D Analytics: CR Prediction Engine (Strict Protocol)")
print("=" * 60)

# 1. CARICAMENTO DATI
# ------------------------------------------------------------------------------
path = "./data/monsters_lean.csv"

df = pd.read_csv(path)
print(f" Dataset caricato da '{path}': {df.shape}")

if df is None:
    print("ERRORE CRITICO: Il file 'monsters_lean.csv' non è stato trovato.")
    print(" Esegui prima lo script 'optimization.ipynb' per generarlo.")
    exit()

# 2. DEFINIZIONE FEATURE (Automatic Detection)
# ------------------------------------------------------------------------------
target = 'cr' if 'cr' in df.columns else 'challenge_rating'
cols_to_exclude = ['name', target, 'challenge_rating', 'Unnamed: 0']

feature_cols = [c for c in df.columns if c not in cols_to_exclude]

print(f" Feature utilizzate ({len(feature_cols)}): {feature_cols}")

# 3. ADDESTRAMENTO MODELLO (STRICT BLINDFOLD)
# ------------------------------------------------------------------------------
# max_features='log2' costringe ogni albero a vedere pochissime feature (4 su 20),
# aumentando drasticamente la probabilità che usi i danni invece degli HP.

print("\n Training Modello 'Strict Blindfold' (log2, 500 alberi)...")

X = df[feature_cols].fillna(0)
y = df[target]

rf_tuned = RandomForestRegressor(
    n_estimators=500,  # Più alberi per compensare la visione limitata
    max_features='log2',
    min_samples_leaf=2,  # Sensibilità agli outlier
    random_state=42,
    n_jobs=-1
)

rf_tuned.fit(X, y)
print(" Training completato.")
print("-" * 60)


# ==============================================================================
# MOTORE DI PREDIZIONE
# ==============================================================================

def predict_monster_cr(model, feature_list, monster_dict):
    """
    Predice il CR di un mostro partendo da un dizionario.
    Calcola automaticamente 'offensive_threat' se mancante.
    """
    data = monster_dict.copy()

    # 1. Calcolo Automatico Threat (Dmg * Actions)
    if 'offensive_threat' not in data:
        dmg = data.get('max_damage_per_hit', 0)
        acts = data.get('actions_count', 1)
        data['offensive_threat'] = dmg * acts

    # 2. Creazione DataFrame e allineamento colonne
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=feature_list, fill_value=0)

    # 3. Predizione
    return model.predict(input_df)[0]


# ==============================================================================
#  SCENARI DI TEST
# ==============================================================================

# SCENARIO A: Il Boss Custom "Lord of Cinder"
# -------------------------------------------------------
print("\n TEST 1: CUSTOM BOSS ('Lord of Cinder')")
lord_of_cinder = {
    'hit_points': 200,
    'armor_class': 19,
    'constitution': 22,
    'hit_dice_count': 20,
    'strength': 24,
    'dexterity': 14,
    'charisma': 20,
    'wisdom_save': 9,
    'constitution_save': 12,
    'damage_immunities_count': 2,
    'condition_immunities_count': 4,
    'actions_count': 3,
    'legendary_actions_count': 3,
    'max_damage_per_hit': 25  # Minaccia: 75 danni/round
}
cr_loc = predict_monster_cr(rf_tuned, feature_cols, lord_of_cinder)
print(f"   HP: {lord_of_cinder['hit_points']} | AC: {lord_of_cinder['armor_class']} | Dmg: 75/rnd")
print(f"   CR Predetto: {cr_loc:.2f}")

# SCENARIO B: Il "Glass Cannon" (L'Assassino)
# -------------------------------------------------------
print("\n  TEST 2: GLASS CANNON CHECK (L'Assassino)")
print("   Obiettivo: Verificare se il danno conta quanto la vita.")

assassin_base = {
    'hit_points': 60,
    'armor_class': 15,
    'dexterity': 20,
    'actions_count': 3,
    'max_damage_per_hit': 30,
    'hit_dice_count': 8,
    'constitution_save': 4
}

cr_full = predict_monster_cr(rf_tuned, feature_cols, assassin_base)

# 2. Nerf Offensivo
assassin_weak_atk = assassin_base.copy()
assassin_weak_atk['max_damage_per_hit'] = 5
assassin_weak_atk['actions_count'] = 1
cr_weak_atk = predict_monster_cr(rf_tuned, feature_cols, assassin_weak_atk)

# 3. Nerf Difensivo
assassin_weak_hp = assassin_base.copy()
assassin_weak_hp['hit_points'] = 30
cr_weak_hp = predict_monster_cr(rf_tuned, feature_cols, assassin_weak_hp)

delta_atk = cr_full - cr_weak_atk
delta_hp = cr_full - cr_weak_hp

print(f"   A. Assassino Letale (60 HP, 90 Dmg): CR {cr_full:.2f}")
print(f"   B. Assassino Inetto (60 HP,  5 Dmg): CR {cr_weak_atk:.2f} (Delta: -{delta_atk:.2f})")
print(f"   C. Assassino Fragile (30 HP, 90 Dmg): CR {cr_weak_hp:.2f} (Delta: -{delta_hp:.2f})")

if delta_atk > 0.4:
    print("RISULTATO: Il modello riconosce l'offensiva! (Delta significativo)")
else:
    print("RISULTATO: Il modello è ancora titubante sui danni.")

# SCENARIO C: Generazione Mostri Bilanciati
# -------------------------------------------------------
print("\n TEST 3: MOSTRI BILANCIATI (Generati per il modello)")
print("(Questi archetipi dovrebbero ottenere CR corretti)")

balanced_roster = [
    {
        "name": "Ironbound Sentinel (Tank)",
        "hit_points": 85, "armor_class": 18, "constitution": 18,
        "actions_count": 2, "max_damage_per_hit": 15,
        "damage_resistances_count": 1, "hit_dice_count": 10
    },
    {
        "name": "Stormborn Goliath (Bruiser)",
        "hit_points": 180, "armor_class": 15, "constitution": 22,
        "actions_count": 3, "max_damage_per_hit": 28,
        "damage_immunities_count": 1, "hit_dice_count": 18
    },
    # --- LICH INCOMPLETO ---
    {
        "name": "Void Lich (incompleto)",
        "hit_points": 210, "armor_class": 19, "constitution": 20,
        "actions_count": 3, "max_damage_per_hit": 45, "legendary_actions_count": 3,
        "damage_immunities_count": 3, "condition_immunities_count": 5
    },
    # --- LICH FULL STATS ---
    {
        "name": "Void Lich (Full Power)",
        # 1. Difesa Fisica
        "hit_points": 210,
        "armor_class": 19,
        "constitution": 20,
        "hit_dice_count": 22,
        "damage_resistances_count": 2,
        "damage_immunities_count": 3,
        "condition_immunities_count": 5,

        "charisma": 20,
        "intelligence_save": 13,
        "wisdom_save": 10,
        "constitution_save": 11,
        "charisma_save": 11,
        "dexterity_save": 5,

        "passive_perception": 22,
        "skills.arcana": 18,
        "skills.stealth": 0,

        "actions_count": 3,
        "max_damage_per_hit": 45,
        "legendary_actions_count": 3,
        "special_abilities_count": 5
    }
]

for m in balanced_roster:
    cr = predict_monster_cr(rf_tuned, feature_cols, m)
    print(f"   🔹 {m['name']:<30} -> CR Stimato: {cr:.2f}")

print("\n" + "=" * 60)
print("  DEMO COMPLETATA.")