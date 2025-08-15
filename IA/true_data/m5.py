# m5_kpis_refactored_fixed.py
"""
KPIs and metrics calculation module for cybersecurity dashboard.
Generates key performance indicators and trend visualizations.
Fixed to handle empty rows in CSV files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


class CyberSecurityKPIs:
    """Calculates and visualizes key performance indicators for cybersecurity metrics."""
    
    def __init__(self, incidents_csv: str = "incidents.csv", logins_csv: str = "logins.csv"):
        self.incidents_path = Path(incidents_csv)
        self.logins_path = Path(logins_csv)
        self.incidents_df = None
        self.logins_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for analysis, handling empty rows."""
        try:
            # Load with empty row handling
            self.incidents_df = pd.read_csv(
                self.incidents_path, 
                parse_dates=["Date"], 
                skip_blank_lines=True
            ).dropna(how='all')
            
            self.logins_df = pd.read_csv(
                self.logins_path, 
                parse_dates=["DateHeure"], 
                skip_blank_lines=True
            ).dropna(how='all')
            
            # Clean critical columns
            if not self.incidents_df.empty:
                self.incidents_df = self.incidents_df.dropna(subset=["Date"])
                # Fill missing impact values
                if "ImpactAriary" in self.incidents_df.columns:
                    self.incidents_df["ImpactAriary"] = self.incidents_df["ImpactAriary"].fillna(0)
            
            if not self.logins_df.empty:
                self.logins_df = self.logins_df.dropna(subset=["DateHeure"])
                # Standardize login results
                if "Resultat" in self.logins_df.columns:
                    self.logins_df["Resultat"] = self.logins_df["Resultat"].fillna("unknown").str.lower().str.strip()
            
            print(f"Données chargées: {len(self.incidents_df)} incidents, {len(self.logins_df)} logins")
            return self.incidents_df, self.logins_df
            
        except FileNotFoundError as e:
            print(f"Fichier non trouvé: {e}")
            self.incidents_df = pd.DataFrame()
            self.logins_df = pd.DataFrame()
            return self.incidents_df, self.logins_df
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            self.incidents_df = pd.DataFrame()
            self.logins_df = pd.DataFrame()
            return self.incidents_df, self.logins_df
    
    def calculate_monthly_incidents(self) -> pd.Series:
        """Calculate incidents per month."""
        if (self.incidents_df is None or self.incidents_df.empty or 
            "Date" not in self.incidents_df.columns):
            print("Aucune donnée d'incident disponible")
            return pd.Series([], dtype=int, name="nb_incidents")
        
        # Filter valid dates
        valid_data = self.incidents_df[self.incidents_df["Date"].notna()]
        if valid_data.empty:
            return pd.Series([], dtype=int, name="nb_incidents")
        
        return valid_data.set_index("Date").resample("ME").size().rename("nb_incidents")
    
    def calculate_quarterly_impact(self) -> pd.Series:
        """Calculate total financial impact per quarter."""
        if (self.incidents_df is None or self.incidents_df.empty or 
            "Date" not in self.incidents_df.columns or 
            "ImpactAriary" not in self.incidents_df.columns):
            print("Données d'impact non disponibles")
            return pd.Series([], dtype=float, name="impact")
        
        # Filter valid data
        valid_data = self.incidents_df[
            self.incidents_df["Date"].notna() & 
            self.incidents_df["ImpactAriary"].notna() &
            (self.incidents_df["ImpactAriary"] >= 0)
        ]
        
        if valid_data.empty:
            return pd.Series([], dtype=float, name="impact")
        
        return valid_data.set_index("Date").resample("QE")["ImpactAriary"].sum()
    
    def calculate_monthly_failure_rate(self) -> pd.Series:
        """Calculate monthly login failure rates."""
        if (self.logins_df is None or self.logins_df.empty or 
            "DateHeure" not in self.logins_df.columns or 
            "Resultat" not in self.logins_df.columns):
            print("Données de login non disponibles")
            return pd.Series([], dtype=float, name="failure_rate")
        
        # Filter valid data
        valid_data = self.logins_df[
            self.logins_df["DateHeure"].notna() & 
            self.logins_df["Resultat"].notna() &
            (self.logins_df["Resultat"] != "unknown")
        ]
        
        if valid_data.empty:
            return pd.Series([], dtype=float, name="failure_rate")
        
        return (valid_data.set_index("DateHeure")
                .resample("ME")["Resultat"]
                .apply(lambda s: (s == "échec").mean()))
    
    def plot_monthly_incidents(self, save_path: Optional[str] = "incidents_par_mois.png") -> None:
        """Plot monthly incidents trend."""
        monthly_incidents = self.calculate_monthly_incidents()
        
        if monthly_incidents.empty:
            print("Pas de données pour tracer les incidents mensuels")
            return
        
        plt.figure(figsize=(12, 6))
        monthly_incidents.plot(marker='o', linewidth=2)
        plt.title("Incidents par mois", fontsize=14, fontweight='bold')
        plt.xlabel("Mois")
        plt.ylabel("Nombre d'incidents")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_quarterly_impact(self, save_path: Optional[str] = "impact_trimestriel.png") -> None:
        """Plot quarterly financial impact trend."""
        quarterly_impact = self.calculate_quarterly_impact()
        
        if quarterly_impact.empty:
            print("Pas de données pour tracer l'impact trimestriel")
            return
        
        plt.figure(figsize=(12, 6))
        quarterly_impact.plot(kind='bar', color='red', alpha=0.7)
        plt.title("Impact financier total par trimestre", fontsize=14, fontweight='bold')
        plt.xlabel("Trimestre")
        plt.ylabel("Impact (MGA)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_monthly_failure_rate(self, save_path: Optional[str] = "taux_echec_mensuel.png") -> None:
        """Plot monthly failure rate trend."""
        monthly_failure_rate = self.calculate_monthly_failure_rate()
        
        if monthly_failure_rate.empty:
            print("Pas de données pour tracer le taux d'échec mensuel")
            return
        
        plt.figure(figsize=(12, 6))
        monthly_failure_rate.plot(marker='s', linewidth=2, color='orange')
        plt.title("Taux d'échec des connexions (mensuel)", fontsize=14, fontweight='bold')
        plt.xlabel("Mois")
        plt.ylabel("Taux d'échec")
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")