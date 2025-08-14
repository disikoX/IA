# m5_kpis_refactored.py
"""
KPIs and metrics calculation module for cybersecurity dashboard.
Generates key performance indicators and trend visualizations.
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
        """Load and prepare data for analysis."""
        self.incidents_df = pd.read_csv(self.incidents_path, parse_dates=["Date"])
        self.logins_df = pd.read_csv(self.logins_path, parse_dates=["DateHeure"])
        
        # Standardize login results
        if "Resultat" in self.logins_df.columns:
            self.logins_df["Resultat"] = self.logins_df["Resultat"].str.lower().str.strip()
        
        return self.incidents_df, self.logins_df
    
    def calculate_monthly_incidents(self) -> pd.Series:
        """Calculate incidents per month."""
        if self.incidents_df is None or "Date" not in self.incidents_df.columns:
            raise ValueError("Incidents data not available or missing Date column")
        
        return self.incidents_df.set_index("Date").resample("ME").size().rename("nb_incidents")
    
    def calculate_quarterly_impact(self) -> pd.Series:
        """Calculate total financial impact per quarter."""
        if (self.incidents_df is None or 
            "Date" not in self.incidents_df.columns or 
            "ImpactAriary" not in self.incidents_df.columns):
            raise ValueError("Incidents data not available or missing required columns")
        
        return self.incidents_df.set_index("Date").resample("QE")["ImpactAriary"].sum()
    
    def calculate_monthly_failure_rate(self) -> pd.Series:
        """Calculate monthly login failure rates."""
        if (self.logins_df is None or 
            "DateHeure" not in self.logins_df.columns or 
            "Resultat" not in self.logins_df.columns):
            raise ValueError("Logins data not available or missing required columns")
        
        return (self.logins_df.set_index("DateHeure")
                .resample("ME")["Resultat"]
                .apply(lambda s: (s == "échec").mean()))
    
    def plot_monthly_incidents(self, save_path: Optional[str] = "incidents_par_mois.png") -> None:
        """Plot monthly incidents trend."""
        monthly_incidents = self.calculate_monthly_incidents()
        
        plt.figure(figsize=(12, 6))
        monthly_incidents.plot(marker='o', linewidth=2)
        plt.title("Incidents par mois", fontsize=14, fontweight='bold')
        plt.xlabel("Mois")
        plt.ylabel("Nombre d'incidents")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def plot_quarterly_impact(self, save_path: Optional[str] = "impact_trimestriel.png") -> None:
        """Plot quarterly financial impact trend."""
        quarterly_impact = self.calculate_quarterly_impact()
        
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
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def plot_monthly_failure_rate(self, save_path: Optional[str] = "taux_echec_mensuel.png") -> None:
        """Plot monthly failure rate trend."""
        monthly_failure_rate = self.calculate_monthly_failure_rate()
        
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
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def compare_periods(self, cutoff_date: str, metric: str = "failure_rate") -> dict:
        """Compare metrics before and after a specific date (e.g., security campaign)."""
        cutoff = pd.to_datetime(cutoff_date)
        
        if self.logins_df is None:
            raise ValueError("Logins data not available")
        
        before_data = self.logins_df[self.logins_df["DateHeure"] < cutoff]
        after_data = self.logins_df[self.logins_df["DateHeure"] >= cutoff]
        
        if metric == "failure_rate":
            before_rate = (before_data["Resultat"] == "échec").mean() if len(before_data) > 0 else 0
            after_rate = (after_data["Resultat"] == "échec").mean() if len(after_data) > 0 else 0
            
            result = {
                "metric": "failure_rate",
                "cutoff_date": cutoff_date,
                "before_period": {
                    "rate": before_rate,
                    "count": len(before_data)
                },
                "after_period": {
                    "rate": after_rate,
                    "count": len(after_data)
                },
                "improvement": before_rate - after_rate,
                "improvement_pct": ((before_rate - after_rate) / before_rate * 100) if before_rate > 0 else 0
            }
            
            print(f"Avant {cutoff_date}: {before_rate:.2%} | Après: {after_rate:.2%}")
            print(f"Amélioration: {result['improvement_pct']:.1f}%")
            
            return result
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def generate_executive_summary(self) -> dict:
        """Generate executive summary with key metrics."""
        if self.incidents_df is None or self.logins_df is None:
            raise ValueError("Both datasets must be loaded")
        
        summary = {
            "period": {
                "start": self.incidents_df["Date"].min().strftime("%Y-%m-%d"),
                "end": self.incidents_df["Date"].max().strftime("%Y-%m-%d")
            },
            "incidents": {
                "total_count": len(self.incidents_df),
                "total_impact": self.incidents_df["ImpactAriary"].sum() if "ImpactAriary" in self.incidents_df.columns else 0,
                "avg_monthly": len(self.incidents_df) / max(1, self.incidents_df["Date"].dt.to_period("M").nunique()),
                "most_affected_sector": self.incidents_df["Secteur"].mode().iloc[0] if "Secteur" in self.incidents_df.columns and not self.incidents_df["Secteur"].empty else "N/A"
            },
            "logins": {
                "total_attempts": len(self.logins_df),
                "failure_rate": (self.logins_df["Resultat"] == "échec").mean() if "Resultat" in self.logins_df.columns else 0,
                "unique_users": self.logins_df["Utilisateur"].nunique() if "Utilisateur" in self.logins_df.columns else 0,
                "unique_ips": self.logins_df["IPSource"].nunique() if "IPSource" in self.logins_df.columns else 0
            }
        }
        
        return summary
    
    def run_complete_kpi_analysis(self, save_charts: bool = True) -> dict:
        """Run complete KPI analysis and generate all visualizations."""
        # Load data
        self.load_data()
        
        # Generate visualizations
        print("Generating KPI visualizations...")
        
        if save_charts:
            self.plot_monthly_incidents()
            self.plot_quarterly_impact()
            self.plot_monthly_failure_rate()
        else:
            self.plot_monthly_incidents(save_path=None)
            self.plot_quarterly_impact(save_path=None)
            self.plot_monthly_failure_rate(save_path=None)
        
        # Compare periods (example with July 1st, 2025)
        try:
            comparison = self.compare_periods("2025-07-01", "failure_rate")
        except Exception as e:
            print(f"Period comparison failed: {e}")
            comparison = None
        
        # Generate executive summary
        summary = self.generate_executive_summary()
        
        return {
            "summary": summary,
            "period_comparison": comparison
        }


def main():
    """Main execution function."""
    kpi_analyzer = CyberSecurityKPIs()
    results = kpi_analyzer.run_complete_kpi_analysis()
    
    print("\n=== EXECUTIVE SUMMARY ===")
    print(f"Period: {results['summary']['period']['start']} to {results['summary']['period']['end']}")
    print(f"Total Incidents: {results['summary']['incidents']['total_count']:,}")
    print(f"Total Impact: {results['summary']['incidents']['total_impact']:,.0f} MGA")
    print(f"Average Monthly Incidents: {results['summary']['incidents']['avg_monthly']:.1f}")
    print(f"Login Failure Rate: {results['summary']['logins']['failure_rate']:.2%}")
    print(f"Unique Users: {results['summary']['logins']['unique_users']:,}")


if __name__ == "__main__":
    main()