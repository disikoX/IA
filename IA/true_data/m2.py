# m2_exploration_refactored.py
"""
Data exploration module for cybersecurity incidents and login data.
Provides data cleaning, analysis, and visualization capabilities.
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
        """Load and return both datasets."""
        self.incidents_df = pd.read_csv(self.incidents_path, parse_dates=["Date"])
        self.logins_df = pd.read_csv(self.logins_path, parse_dates=["DateHeure"])
        return self.incidents_df, self.logins_df
    
    @staticmethod
    def clean_incidents(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize incidents data."""
        df = df.copy()
        
        # Text normalization
        text_columns = {
            "Secteur": lambda x: x.str.strip().str.title(),
            "TypeAttaque": lambda x: x.str.strip().str.lower(),
            "Vecteur": lambda x: x.str.strip().str.lower(),
            "Entreprise": lambda x: x.str.strip()
        }
        
        for col, transform in text_columns.items():
            if col in df.columns:
                df[col] = transform(df[col])
        
        # Handle negative or invalid values
        numeric_columns = ["ImpactAriary", "IndispoHeures"]
        for col in numeric_columns:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan
        
        return df
    
    @staticmethod
    def clean_logins(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize login data."""
        df = df.copy()
        
        # Normalize result column
        if "Resultat" in df.columns:
            df["Resultat"] = df["Resultat"].str.lower().str.strip()
        
        # Fill missing values and normalize
        if "Localisation" in df.columns:
            df["Localisation"] = df["Localisation"].fillna("Unknown").str.title()
        
        if "Role" in df.columns:
            df["Role"] = df["Role"].fillna("Employe").str.title()
        
        return df
    
    def display_summary_stats(self) -> None:
        """Display summary statistics for both datasets."""
        if self.incidents_df is not None:
            print("=== INCIDENTS: Aperçu ===")
            print(self.incidents_df.head(10))
            print("\n=== INCIDENTS: Stats clés ===")
            numeric_cols = ["ImpactAriary", "IndispoHeures", "Taille"]
            available_cols = [col for col in numeric_cols if col in self.incidents_df.columns]
            if available_cols:
                print(self.incidents_df[available_cols].describe())
        
        if self.logins_df is not None:
            print("\n=== LOGINS: Aperçu ===")
            print(self.logins_df.head(10))
            print("\n=== LOGINS: Volume par résultat ===")
            if "Resultat" in self.logins_df.columns:
                print(self.logins_df["Resultat"].value_counts())
    
    def plot_incidents_by_type(self) -> None:
        """Plot incidents by attack type."""
        if self.incidents_df is None or "TypeAttaque" not in self.incidents_df.columns:
            print("Incidents data not available or missing TypeAttaque column")
            return
        
        counts_type = self.incidents_df["TypeAttaque"].value_counts().sort_values(ascending=False)
        
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
        if (self.incidents_df is None or 
            "Secteur" not in self.incidents_df.columns or 
            "ImpactAriary" not in self.incidents_df.columns):
            print("Incidents data not available or missing required columns")
            return
        
        impact_secteur = (self.incidents_df.groupby("Secteur")["ImpactAriary"]
                         .median()
                         .sort_values(ascending=False))
        
        plt.figure(figsize=(10, 6))
        impact_secteur.plot(kind="bar")
        plt.title("Médiane de l'impact (€) par secteur")
        plt.xlabel("Secteur")
        plt.ylabel("Impact (€) médian")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_daily_login_attempts(self) -> None:
        """Plot daily login attempts volume."""
        if self.logins_df is None or "DateHeure" not in self.logins_df.columns:
            print("Logins data not available or missing DateHeure column")
            return
        
        log_day = self.logins_df.set_index("DateHeure").resample("D").size()
        
        plt.figure(figsize=(12, 6))
        log_day.plot()
        plt.title("Volume de tentatives de connexion par jour")
        plt.xlabel("Date")
        plt.ylabel("Nombre de tentatives")
        plt.tight_layout()
        plt.show()
    
    def calculate_failure_rate(self) -> float:
        """Calculate global login failure rate."""
        if self.logins_df is None or "Resultat" not in self.logins_df.columns:
            print("Logins data not available or missing Resultat column")
            return 0.0
        
        failure_rate = (self.logins_df["Resultat"] == "échec").mean()
        print(f"\nTaux d'échec global des connexions: {failure_rate:.2%}")
        return failure_rate
    
    def run_complete_analysis(self) -> None:
        """Run the complete data exploration analysis."""
        # Load and clean data
        inc_df, log_df = self.load_data()
        self.incidents_df = self.clean_incidents(inc_df)
        self.logins_df = self.clean_logins(log_df)
        
        # Display summary statistics
        self.display_summary_stats()
        
        # Generate visualizations
        self.plot_incidents_by_type()
        self.plot_impact_by_sector()
        self.plot_daily_login_attempts()
        
        # Calculate metrics
        self.calculate_failure_rate()


def main():
    """Main execution function."""
    explorer = CyberSecurityDataExplorer()
    explorer.run_complete_analysis()


if __name__ == "__main__":
    main()