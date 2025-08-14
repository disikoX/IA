# m4_profiling_refactored.py
"""
Profiling module for analyzing enterprise and user segments.
Provides detailed cluster analysis and characterization.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


class SegmentProfiler:
    """Analyzes and profiles customer and user segments."""
    
    def __init__(self, enterprises_csv: str = "segments_entreprises.csv", 
                 users_csv: str = "segments_utilisateurs.csv"):
        self.enterprises_path = Path(enterprises_csv)
        self.users_path = Path(users_csv)
        
    def load_segments(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load enterprise and user segment data."""
        try:
            enterprises = pd.read_csv(self.enterprises_path)
            users = pd.read_csv(self.users_path)
            return enterprises, users
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Segment files not found: {e}")
    
    def analyze_enterprise_cluster(self, df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Analyze a specific enterprise cluster."""
        cluster_data = df[df["cluster_esn"] == cluster_id]
        
        if cluster_data.empty:
            return {"error": f"No data found for cluster {cluster_id}"}
        
        analysis = {
            "cluster_id": cluster_id,
            "size": len(cluster_data),
            "dominant_sectors": self._get_top_values(cluster_data, "secteur", 3),
            "median_size": int(cluster_data["taille"].median()) if "taille" in cluster_data.columns else "N/A",
            "median_incident_freq": round(cluster_data["freq_incidents"].median(), 2) if "freq_incidents" in cluster_data.columns else "N/A",
            "avg_impact": round(cluster_data["impact_moy"].mean(), 2) if "impact_moy" in cluster_data.columns else "N/A",
            "avg_downtime": round(cluster_data["indispo_moy"].mean(), 2) if "indispo_moy" in cluster_data.columns else "N/A",
            "avg_attack_diversity": round(cluster_data["nb_types"].mean(), 2) if "nb_types" in cluster_data.columns else "N/A"
        }
        
        return analysis
    
    def analyze_user_cluster(self, df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Analyze a specific user cluster."""
        cluster_data = df[df["cluster_risque"] == cluster_id]
        
        if cluster_data.empty:
            return {"error": f"No data found for cluster {cluster_id}"}
        
        analysis = {
            "cluster_id": cluster_id,
            "size": len(cluster_data),
            "dominant_roles": self._get_top_values(cluster_data, "Role", 3),
            "dominant_departments": self._get_top_values(cluster_data, "Departement", 3),
            "median_failures": int(cluster_data["nb_echecs"].median()) if "nb_echecs" in cluster_data.columns else "N/A",
            "avg_failure_ratio": round(cluster_data["ratio_echec"].mean(), 3) if "ratio_echec" in cluster_data.columns else "N/A",
            "avg_countries": round(cluster_data["nb_pays"].mean(), 2) if "nb_pays" in cluster_data.columns else "N/A",
            "avg_ips": round(cluster_data["nb_ip"].mean(), 2) if "nb_ip" in cluster_data.columns else "N/A"
        }
        
        return analysis
    
    def _get_top_values(self, df: pd.DataFrame, column: str, n: int = 3) -> Dict[str, int]:
        """Get top n values from a column as dictionary."""
        if column not in df.columns:
            return {"N/A": 0}
        
        return df[column].value_counts().head(n).to_dict()
    
    def profile_enterprises(self, df: pd.DataFrame) -> None:
        """Profile all enterprise clusters."""
        print("=== PROFILAGE ENTREPRISES ===")
        
        if "cluster_esn" not in df.columns:
            print("No cluster_esn column found in enterprise data")
            return
        
        clusters = sorted(df["cluster_esn"].unique())
        
        for cluster_id in clusters:
            analysis = self.analyze_enterprise_cluster(df, cluster_id)
            
            if "error" in analysis:
                print(f"\n{analysis['error']}")
                continue
            
            print(f"\nCluster {analysis['cluster_id']} — {analysis['size']} entreprises")
            print("Secteurs dominants:", analysis["dominant_sectors"])
            print("Taille médiane:", analysis["median_size"])
            print("Freq incidents médian:", analysis["median_incident_freq"])
            print("Impact moyen:", analysis["avg_impact"])
            print("Indispo moyenne:", analysis["avg_downtime"])
            print("Diversité attaques (moy):", analysis["avg_attack_diversity"])
    
    def profile_users(self, df: pd.DataFrame) -> None:
        """Profile all user clusters."""
        print("\n=== PROFILAGE UTILISATEURS ===")
        
        if "cluster_risque" not in df.columns:
            print("No cluster_risque column found in user data")
            return
        
        clusters = sorted(df["cluster_risque"].unique())
        
        for cluster_id in clusters:
            analysis = self.analyze_user_cluster(df, cluster_id)
            
            if "error" in analysis:
                print(f"\n{analysis['error']}")
                continue
            
            print(f"\nCluster {analysis['cluster_id']} — {analysis['size']} utilisateurs")
            print("Rôles dominants:", analysis["dominant_roles"])
            print("Départements dominants:", analysis["dominant_departments"])
            print("Nb échecs médian:", analysis["median_failures"])
            print("Ratio échec moyen:", analysis["avg_failure_ratio"])
            print("Pays distincts moyen:", analysis["avg_countries"])
            print("IP distinctes moyen:", analysis["avg_ips"])
    
    def generate_cluster_summary(self, df: pd.DataFrame, cluster_col: str, cluster_type: str) -> pd.DataFrame:
        """Generate a summary DataFrame for all clusters."""
        clusters = sorted(df[cluster_col].unique())
        summaries = []
        
        for cluster_id in clusters:
            cluster_data = df[df[cluster_col] == cluster_id]
            
            summary = {
                "cluster_id": cluster_id,
                "cluster_type": cluster_type,
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(df) * 100
            }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def run_complete_profiling(self) -> None:
        """Run complete profiling analysis."""
        try:
            enterprises, users = self.load_segments()
            
            # Profile segments
            self.profile_enterprises(enterprises)
            self.profile_users(users)
            
            # Generate summaries
            if "cluster_esn" in enterprises.columns:
                ent_summary = self.generate_cluster_summary(enterprises, "cluster_esn", "enterprise")
                print("\n=== Enterprise Cluster Summary ===")
                print(ent_summary)
            
            if "cluster_risque" in users.columns:
                user_summary = self.generate_cluster_summary(users, "cluster_risque", "user")
                print("\n=== User Cluster Summary ===")
                print(user_summary)
                
        except Exception as e:
            print(f"Error during profiling: {e}")


def main():
    """Main execution function."""
    profiler = SegmentProfiler()
    profiler.run_complete_profiling()


if __name__ == "__main__":
    main()