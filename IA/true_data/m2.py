# m2_exploration_refactored_fixed.py
"""
Data exploration module for cybersecurity incidents and login data.
Provides data cleaning, analysis, and visualization capabilities.
Fixed to handle empty rows in CSV files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


class CyberSecurityDataExplorer:
    """Handles exploration and analysis of cybersecurity data."""
    
    def __init__(self, incidents_csv: str = "incidents.csv", logins_csv: str = "logins.csv"):
        self.incidents_path = Path(incidents_csv)
        self.logins_path = Path(logins_csv)
        self.incidents_df = None
        self.logins_df = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and return both datasets, handling empty rows."""
        # Load with skipna and drop empty rows
        self.incidents_df = pd.read_csv(
            self.incidents_path, 
            parse_dates=["Date"],
            skip_blank_lines=True
        ).dropna(how='all')  # Drop completely empty rows
        
        self.logins_df = pd.read_csv(
            self.logins_path, 
            parse_dates=["DateHeure"],
            skip_blank_lines=True
        ).dropna(how='all')  # Drop completely empty rows
        
        # Remove rows where critical columns are empty
        if not self.incidents_df.empty and "Date" in self.incidents_df.columns:
            self.incidents_df = self.incidents_df.dropna(subset=["Date"])
        
        if not self.logins_df.empty and "DateHeure" in self.logins_df.columns:
            self.logins_df = self.logins_df.dropna(subset=["DateHeure"])
        
        print(f"Loaded {len(self.incidents_df)} incidents and {len(self.logins_df)} login records")
        
        return self.incidents_df, self.logins_df
    
    @staticmethod
    def clean_incidents(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize incidents data."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Text normalization - handle NaN values
        text_columns = {
            "Secteur": lambda x: x.fillna("Unknown").str.strip().str.title(),
            "TypeAttaque": lambda x: x.fillna("unknown").str.strip().str.lower(),
            "Vecteur": lambda x: x.fillna("unknown").str.strip().str.lower(),
            "Entreprise": lambda x: x.fillna("Unknown").str.strip()
        }
        
        for col, transform in text_columns.items():
            if col in df.columns:
                df[col] = transform(df[col])
        
        # Handle negative or invalid values
        numeric_columns = ["ImpactAriary", "IndispoHeures", "Taille"]
        for col in numeric_columns:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan
                # Fill NaN with median for numeric columns
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    @staticmethod
    def clean_logins(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize login data."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Normalize result column - handle NaN values
        if "Resultat" in df.columns:
            df["Resultat"] = df["Resultat"].fillna("unknown").str.lower().str.strip()
        
        # Fill missing values and normalize
        if "Localisation" in df.columns:
            df["Localisation"] = df["Localisation"].fillna("Unknown").str.title()
        
        if "Role" in df.columns:
            df["Role"] = df["Role"].fillna("Employe").str.title()
        
        if "Utilisateur" in df.columns:
            df["Utilisateur"] = df["Utilisateur"].fillna("Unknown")
        
        if "IPSource" in df.columns:
            df["IPSource"] = df["IPSource"].fillna("0.0.0.0")
        
        return df
    
    def display_summary_stats(self) -> None:
        """Display summary statistics for both datasets."""
        if self.incidents_df is not None and not self.incidents_df.empty:
            print("=== INCIDENTS: Aperçu ===")
            print(self.incidents_df.head(10))
            print(f"\nNombre total d'incidents: {len(self.incidents_df)}")
            
            # Check for missing values
            missing_counts = self.incidents_df.isnull().sum()
            if missing_counts.sum() > 0:
                print("\n=== INCIDENTS: Valeurs manquantes ===")
                print(missing_counts[missing_counts > 0])
            
            print("\n=== INCIDENTS: Stats clés ===")
            numeric_cols = ["ImpactAriary", "IndispoHeures", "Taille"]
            available_cols = [col for col in numeric_cols if col in self.incidents_df.columns]
            if available_cols:
                print(self.incidents_df[available_cols].describe())
        else:
            print("Aucune donnée d'incident disponible ou fichier vide")
        
        if self.logins_df is not None and not self.logins_df.empty:
            print("\n=== LOGINS: Aperçu ===")
            print(self.logins_df.head(10))
            print(f"\nNombre total de tentatives de connexion: {len(self.logins_df)}")
            
            # Check for missing values
            missing_counts = self.logins_df.isnull().sum()
            if missing_counts.sum() > 0:
                print("\n=== LOGINS: Valeurs manquantes ===")
                print(missing_counts[missing_counts > 0])
            
            print("\n=== LOGINS: Volume par résultat ===")
            if "Resultat" in self.logins_df.columns:
                print(self.logins_df["Resultat"].value_counts())
        else:
            print("Aucune donnée de login disponible ou fichier vide")
    
    def plot_incidents_by_type(self) -> None:
        """Plot incidents by attack type."""
        if (self.incidents_df is None or self.incidents_df.empty or 
            "TypeAttaque" not in self.incidents_df.columns):
            print("Données d'incidents non disponibles ou colonne TypeAttaque manquante")
            return
        
        # Filter out unknown/empty values for visualization
        valid_data = self.incidents_df[
            self.incidents_df["TypeAttaque"].notna() & 
            (self.incidents_df["TypeAttaque"] != "unknown")
        ]
        
        if valid_data.empty:
            print("Aucune donnée valide pour les types d'attaque")
            return
        
        counts_type = valid_data["TypeAttaque"].value_counts().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        counts_type.plot(kind="bar")
        plt.title("Incidents par type d'attaque")
        plt.xlabel("Type d'attaque")
        plt.ylabel("Nombre d'incidents")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_impact_by_sector(self) -> None:
        """Plot median financial impact by sector."""
        if (self.incidents_df is None or self.incidents_df.empty or
            "Secteur" not in self.incidents_df.columns or 
            "ImpactAriary" not in self.incidents_df.columns):
            print("Données d'incidents non disponibles ou colonnes requises manquantes")
            return
        
        # Filter out rows with missing or invalid data
        valid_data = self.incidents_df[
            self.incidents_df["Secteur"].notna() & 
            self.incidents_df["ImpactAriary"].notna() &
            (self.incidents_df["Secteur"] != "Unknown") &
            (self.incidents_df["ImpactAriary"] > 0)
        ]
        
        if valid_data.empty:
            print("Aucune donnée valide pour l'impact par secteur")
            return
        
        impact_secteur = (valid_data.groupby("Secteur")["ImpactAriary"]
                         .median()
                         .sort_values(ascending=False))
        
        plt.figure(figsize=(10, 6))
        impact_secteur.plot(kind="bar")
        plt.title("Médiane de l'impact (MGA) par secteur")
        plt.xlabel("Secteur")
        plt.ylabel("Impact (MGA) médian")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_daily_login_attempts(self) -> None:
        """Plot daily login attempts volume."""
        if (self.logins_df is None or self.logins_df.empty or 
            "DateHeure" not in self.logins_df.columns):
            print("Données de login non disponibles ou colonne DateHeure manquante")
            return
        
        # Filter out rows with invalid dates
        valid_data = self.logins_df[self.logins_df["DateHeure"].notna()]
        
        if valid_data.empty:
            print("Aucune donnée valide pour les tentatives de connexion")
            return
        
        log_day = valid_data.set_index("DateHeure").resample("D").size()
        
        plt.figure(figsize=(12, 6))
        log_day.plot()
        plt.title("Volume de tentatives de connexion par jour")
        plt.xlabel("Date")
        plt.ylabel("Nombre de tentatives")
        plt.tight_layout()
        plt.show()
    
    def calculate_failure_rate(self) -> float:
        """Calculate global login failure rate."""
        if (self.logins_df is None or self.logins_df.empty or 
            "Resultat" not in self.logins_df.columns):
            print("Données de login non disponibles ou colonne Resultat manquante")
            return 0.0
        
        # Filter out unknown/invalid results
        valid_data = self.logins_df[
            self.logins_df["Resultat"].notna() & 
            (self.logins_df["Resultat"] != "unknown")
        ]
        
        if valid_data.empty:
            print("Aucune donnée valide pour calculer le taux d'échec")
            return 0.0
        
        failure_rate = (valid_data["Resultat"] == "échec").mean()
        print(f"\nTaux d'échec global des connexions: {failure_rate:.2%}")
        print(f"Basé sur {len(valid_data)} tentatives valides")
        return failure_rate
    
    def run_complete_analysis(self) -> None:
        """Run the complete data exploration analysis."""
        try:
            # Load and clean data
            inc_df, log_df = self.load_data()
            self.incidents_df = self.clean_incidents(inc_df)
            self.logins_df = self.clean_logins(log_df)
            
            # Display summary statistics
            self.display_summary_stats()
            
            # Generate visualizations only if data is available
            if self.incidents_df is not None and not self.incidents_df.empty:
                self.plot_incidents_by_type()
                self.plot_impact_by_sector()
            
            if self.logins_df is not None and not self.logins_df.empty:
                self.plot_daily_login_attempts()
                self.calculate_failure_rate()
                
        except Exception as e:
            print(f"Erreur lors de l'analyse: {e}")
            print("Vérifiez que les fichiers CSV existent et contiennent des données valides")


def main():
    """Main execution function."""
    explorer = CyberSecurityDataExplorer()
    explorer.run_complete_analysis()


if __name__ == "__main__":
    main()