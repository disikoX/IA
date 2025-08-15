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
        
        plt.show()
    
    def plot_incident_severity_distribution(self, save_path: Optional[str] = "distribution_severite.png") -> None:
        """Plot distribution of incident severity levels."""
        if (self.incidents_df is None or self.incidents_df.empty or 
            "Severite" not in self.incidents_df.columns):
            print("Données de sévérité non disponibles")
            return
        
        severity_data = self.incidents_df[self.incidents_df["Severite"].notna()]
        if severity_data.empty:
            print("Aucune donnée de sévérité valide")
            return
        
        severity_counts = severity_data["Severite"].value_counts()
        
        plt.figure(figsize=(10, 6))
        colors = ['#ff4444', '#ff8800', '#ffdd00', '#88dd00', '#0088dd']
        severity_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors[:len(severity_counts)])
        plt.title("Distribution des incidents par niveau de sévérité", fontsize=14, fontweight='bold')
        plt.ylabel("")  # Remove default ylabel for pie charts
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_top_attack_vectors(self, top_n: int = 10, save_path: Optional[str] = "top_vecteurs_attaque.png") -> None:
        """Plot top attack vectors."""
        if (self.incidents_df is None or self.incidents_df.empty or 
            "VecteurAttaque" not in self.incidents_df.columns):
            print("Données de vecteur d'attaque non disponibles")
            return
        
        vector_data = self.incidents_df[self.incidents_df["VecteurAttaque"].notna()]
        if vector_data.empty:
            print("Aucune donnée de vecteur d'attaque valide")
            return
        
        top_vectors = vector_data["VecteurAttaque"].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        top_vectors.plot(kind='barh', color='steelblue')
        plt.title(f"Top {top_n} des vecteurs d'attaque", fontsize=14, fontweight='bold')
        plt.xlabel("Nombre d'incidents")
        plt.ylabel("Vecteur d'attaque")
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def calculate_mttr(self) -> dict:
        """Calculate Mean Time To Resolution (MTTR)."""
        if (self.incidents_df is None or self.incidents_df.empty or 
            "Date" not in self.incidents_df.columns or 
            "DateResolution" not in self.incidents_df.columns):
            print("Données de résolution non disponibles pour MTTR")
            return {"error": "Missing resolution data"}
        
        resolved_incidents = self.incidents_df[
            self.incidents_df["Date"].notna() & 
            self.incidents_df["DateResolution"].notna()
        ].copy()
        
        if resolved_incidents.empty:
            return {"error": "No resolved incidents found"}
        
        # Calculate resolution time
        resolved_incidents["ResolutionTime"] = (
            resolved_incidents["DateResolution"] - resolved_incidents["Date"]
        ).dt.total_seconds() / 3600  # Convert to hours
        
        # Filter out negative values (data quality issues)
        valid_resolution_times = resolved_incidents[
            resolved_incidents["ResolutionTime"] >= 0
        ]["ResolutionTime"]
        
        if valid_resolution_times.empty:
            return {"error": "No valid resolution times"}
        
        return {
            "mttr_hours": valid_resolution_times.mean(),
            "median_hours": valid_resolution_times.median(),
            "min_hours": valid_resolution_times.min(),
            "max_hours": valid_resolution_times.max(),
            "std_hours": valid_resolution_times.std(),
            "total_resolved": len(valid_resolution_times)
        }
    
    def generate_trend_analysis(self, window_size: int = 3) -> dict:
        """Generate trend analysis for key metrics."""
        results = {}
        
        # Incidents trend
        monthly_incidents = self.calculate_monthly_incidents()
        if not monthly_incidents.empty and len(monthly_incidents) >= window_size:
            recent_avg = monthly_incidents.tail(window_size).mean()
            previous_avg = monthly_incidents.head(-window_size).mean() if len(monthly_incidents) > window_size else monthly_incidents.mean()
            
            results["incidents_trend"] = {
                "recent_average": recent_avg,
                "previous_average": previous_avg,
                "trend_direction": "increasing" if recent_avg > previous_avg else "decreasing",
                "change_percent": ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            }
        
        # Failure rate trend
        monthly_failure = self.calculate_monthly_failure_rate()
        if not monthly_failure.empty and len(monthly_failure) >= window_size:
            recent_failure = monthly_failure.tail(window_size).mean()
            previous_failure = monthly_failure.head(-window_size).mean() if len(monthly_failure) > window_size else monthly_failure.mean()
            
            results["failure_rate_trend"] = {
                "recent_average": recent_failure,
                "previous_average": previous_failure,
                "trend_direction": "increasing" if recent_failure > previous_failure else "decreasing",
                "change_percent": ((recent_failure - previous_failure) / previous_failure * 100) if previous_failure > 0 else 0
            }
        
        return results
    
    def compare_periods(self, cutoff_date: str, metric: str = "failure_rate") -> dict:
        """Compare metrics before and after a specific date (e.g., security campaign)."""
        cutoff = pd.to_datetime(cutoff_date)
        
        if self.logins_df is None or self.logins_df.empty:
            print("Aucune donnée de login disponible")
            return {"error": "No login data available"}
        
        # Filter valid data
        valid_data = self.logins_df[
            self.logins_df["DateHeure"].notna() & 
            self.logins_df["Resultat"].notna() &
            (self.logins_df["Resultat"] != "unknown")
        ]
        
        if valid_data.empty:
            print("Aucune donnée valide pour la comparaison")
            return {"error": "No valid data for comparison"}
        
        before_data = valid_data[valid_data["DateHeure"] < cutoff]
        after_data = valid_data[valid_data["DateHeure"] >= cutoff]
        
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
            raise ValueError(f"Métrique non supportée: {metric}")
    
    def generate_executive_summary(self) -> dict:
        """Generate executive summary with key metrics."""
        summary = {
            "period": {"start": "N/A", "end": "N/A"},
            "incidents": {
                "total_count": 0,
                "total_impact": 0,
                "avg_monthly": 0,
                "most_affected_sector": "N/A",
                "avg_resolution_time": "N/A"
            },
            "logins": {
                "total_attempts": 0,
                "failure_rate": 0,
                "unique_users": 0,
                "unique_ips": 0
            },
            "trends": {}
        }
        
        # Process incidents data
        if self.incidents_df is not None and not self.incidents_df.empty:
            valid_incidents = self.incidents_df[self.incidents_df["Date"].notna()]
            if not valid_incidents.empty:
                summary["period"]["start"] = valid_incidents["Date"].min().strftime("%Y-%m-%d")
                summary["period"]["end"] = valid_incidents["Date"].max().strftime("%Y-%m-%d")
                summary["incidents"]["total_count"] = len(valid_incidents)
                
                if "ImpactAriary" in valid_incidents.columns:
                    impact_data = valid_incidents[valid_incidents["ImpactAriary"].notna()]
                    summary["incidents"]["total_impact"] = impact_data["ImpactAriary"].sum()
                
                summary["incidents"]["avg_monthly"] = len(valid_incidents) / max(1, valid_incidents["Date"].dt.to_period("M").nunique())
                
                if "Secteur" in valid_incidents.columns:
                    sector_data = valid_incidents[valid_incidents["Secteur"].notna()]
                    if not sector_data.empty:
                        summary["incidents"]["most_affected_sector"] = sector_data["Secteur"].mode().iloc[0]
                
                # Add MTTR
                mttr_data = self.calculate_mttr()
                if "error" not in mttr_data:
                    summary["incidents"]["avg_resolution_time"] = f"{mttr_data['mttr_hours']:.1f} heures"
        
        # Process logins data
        if self.logins_df is not None and not self.logins_df.empty:
            valid_logins = self.logins_df[self.logins_df["DateHeure"].notna()]
            if not valid_logins.empty:
                summary["logins"]["total_attempts"] = len(valid_logins)
                
                if "Resultat" in valid_logins.columns:
                    result_data = valid_logins[
                        valid_logins["Resultat"].notna() & 
                        (valid_logins["Resultat"] != "unknown")
                    ]
                    if not result_data.empty:
                        summary["logins"]["failure_rate"] = (result_data["Resultat"] == "échec").mean()
                
                if "Utilisateur" in valid_logins.columns:
                    user_data = valid_logins[valid_logins["Utilisateur"].notna()]
                    summary["logins"]["unique_users"] = user_data["Utilisateur"].nunique()
                
                if "IPSource" in valid_logins.columns:
                    ip_data = valid_logins[valid_logins["IPSource"].notna()]
                    summary["logins"]["unique_ips"] = ip_data["IPSource"].nunique()
        
        # Add trend analysis
        summary["trends"] = self.generate_trend_analysis()
        
        return summary
    
    def export_kpi_report(self, filename: str = "rapport_kpi_cybersecurite.txt") -> None:
        """Export comprehensive KPI report to text file."""
        summary = self.generate_executive_summary()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAPPORT KPI CYBERSÉCURITÉ\n")
            f.write("=" * 60 + "\n")
            f.write(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Period
            f.write(f"PÉRIODE D'ANALYSE: {summary['period']['start']} à {summary['period']['end']}\n\n")
            
            # Incidents
            f.write("INCIDENTS DE SÉCURITÉ\n")
            f.write("-" * 30 + "\n")
            f.write(f"• Total des incidents: {summary['incidents']['total_count']:,}\n")
            f.write(f"• Impact financier total: {summary['incidents']['total_impact']:,.0f} MGA\n")
            f.write(f"• Moyenne mensuelle: {summary['incidents']['avg_monthly']:.1f} incidents\n")
            f.write(f"• Secteur le plus affecté: {summary['incidents']['most_affected_sector']}\n")
            f.write(f"• Temps moyen de résolution: {summary['incidents']['avg_resolution_time']}\n\n")
            
            # Logins
            f.write("AUTHENTIFICATIONS\n")
            f.write("-" * 30 + "\n")
            f.write(f"• Total des tentatives: {summary['logins']['total_attempts']:,}\n")
            f.write(f"• Taux d'échec: {summary['logins']['failure_rate']:.2%}\n")
            f.write(f"• Utilisateurs uniques: {summary['logins']['unique_users']:,}\n")
            f.write(f"• Adresses IP uniques: {summary['logins']['unique_ips']:,}\n\n")
            
            # Trends
            if summary['trends']:
                f.write("ANALYSE DES TENDANCES\n")
                f.write("-" * 30 + "\n")
                
                if "incidents_trend" in summary['trends']:
                    trend = summary['trends']['incidents_trend']
                    f.write(f"• Incidents - Tendance: {trend['trend_direction']}\n")
                    f.write(f"  Changement: {trend['change_percent']:.1f}%\n")
                
                if "failure_rate_trend" in summary['trends']:
                    trend = summary['trends']['failure_rate_trend']
                    f.write(f"• Taux d'échec - Tendance: {trend['trend_direction']}\n")
                    f.write(f"  Changement: {trend['change_percent']:.1f}%\n")
        
        print(f"Rapport exporté vers: {filename}")
    
    def run_complete_kpi_analysis(self, save_charts: bool = True) -> dict:
        """Run complete KPI analysis and generate all visualizations."""
        # Load data
        self.load_data()
        
        if self.incidents_df.empty and self.logins_df.empty:
            print("Aucune donnée disponible pour l'analyse")
            return {"error": "No data available"}
        
        # Generate visualizations
        print("Génération des visualisations KPI...")
        
        if save_charts:
            self.plot_monthly_incidents()
            self.plot_quarterly_impact()
            self.plot_monthly_failure_rate()
            self.plot_incident_severity_distribution()
            self.plot_top_attack_vectors()
        else:
            self.plot_monthly_incidents(save_path=None)
            self.plot_quarterly_impact(save_path=None)
            self.plot_monthly_failure_rate(save_path=None)
            self.plot_incident_severity_distribution(save_path=None)
            self.plot_top_attack_vectors(save_path=None)
        
        # Compare periods (example with July 1st, 2025)
        try:
            comparison = self.compare_periods("2025-07-01", "failure_rate")
        except Exception as e:
            print(f"Comparaison de périodes échouée: {e}")
            comparison = None
        
        # Generate executive summary
        summary = self.generate_executive_summary()
        
        # Export report
        self.export_kpi_report()
        
        return {
            "summary": summary,
            "period_comparison": comparison,
            "mttr": self.calculate_mttr()
        }


def main():
    """Main execution function."""
    kpi_analyzer = CyberSecurityKPIs()
    results = kpi_analyzer.run_complete_kpi_analysis()
    
    if "error" not in results:
        print("\n" + "=" * 50)
        print("RÉSUMÉ EXÉCUTIF")
        print("=" * 50)
        print(f"Période: {results['summary']['period']['start']} à {results['summary']['period']['end']}")
        print(f"Total Incidents: {results['summary']['incidents']['total_count']:,}")
        print(f"Impact Total: {results['summary']['incidents']['total_impact']:,.0f} MGA")
        print(f"Incidents Mensuels (moyenne): {results['summary']['incidents']['avg_monthly']:.1f}")
        print(f"Taux d'Échec Login: {results['summary']['logins']['failure_rate']:.2%}")
        print(f"Utilisateurs Uniques: {results['summary']['logins']['unique_users']:,}")
        
        if "error" not in results['mttr']:
            print(f"MTTR: {results['mttr']['mttr_hours']:.1f} heures")
        
        # Display trends
        if results['summary']['trends']:
            print("\nTENDANCES:")
            for key, trend in results['summary']['trends'].items():
                if isinstance(trend, dict):
                    direction = "📈" if trend['trend_direction'] == "increasing" else "📉"
                    print(f"{key.replace('_', ' ').title()}: {direction} {trend['change_percent']:.1f}%")
        
    else:
        print("Analyse impossible à effectuer en raison de l'absence de données")


if __name__ == "__main__":
    main()