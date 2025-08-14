# m4_profiling_refactored_fixed.py
"""
Profiling module for analyzing enterprise and user segments.
Provides detailed cluster analysis and characterization.
Fixed to handle empty rows and missing files.
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
        """Load enterprise and user segment data with error handling."""
        def safe_load_csv(filepath):
            try:
                if filepath.exists():
                    df = pd.read_csv(filepath, skip_blank_lines=True).dropna(how='all')
                    return df if not df.empty else pd.DataFrame()
                else:
                    print(f"Fichier non trouvé: {filepath}")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Erreur lors du chargement de {filepath}: {e}")
                return pd.DataFrame()
        
        enterprises = safe_load_csv(self.enterprises_path)
        users = safe_load_csv(self.users_path)
        
        print(f"Chargé: {len(enterprises)} entreprises, {len(users)} utilisateurs")
        return enterprises, users
    
    def analyze_enterprise_cluster(self, df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Analyze a specific enterprise cluster."""
        if df.empty or "cluster_esn" not in df.columns:
            return {"error": "No enterprise cluster data available"}
        
        cluster_data = df[df["cluster_esn"] == cluster_id]
        
        if cluster_data.empty:
            return {"error": f"No data found for cluster {cluster_id}"}
        
        analysis = {
            "cluster_id": cluster_id,
            "size": len(cluster_data),
            "dominant_sectors": self._get_top_values(cluster_data, "secteur", 3),
            "median_size": self._safe_median(cluster_data, "taille"),
            "median_incident_freq": self._safe_median(cluster_data, "freq_incidents"),
            "avg_impact": self._safe_mean(cluster_data, "impact_moy"),
            "avg_downtime": self._safe_mean(cluster_data, "indispo_moy"),
            "avg_attack_diversity": self._safe_mean(cluster_data, "nb_types")
        }
        
        return analysis
    
    def analyze_user_cluster(self, df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Analyze a specific user cluster."""
        if df.empty or "cluster_risque" not in df.columns:
            return {"error": "No user cluster data available"}
        
        cluster_data = df[df["cluster_risque"] == cluster_id]
        
        if cluster_data.empty:
            return {"error": f"No data found for cluster {cluster_id}"}
        
        analysis = {
            "cluster_id": cluster_id,
            "size": len(cluster_data),
            "dominant_roles": self._get_top_values(cluster_data, "Role", 3),
            "dominant_departments": self._get_top_values(cluster_data, "Departement", 3),
            "median_failures": self._safe_median(cluster_data, "nb_echecs"),
            "avg_failure_ratio": self._safe_mean(cluster_data, "ratio_echec"),
            "avg_countries": self._safe_mean(cluster_data, "nb_pays"),
            "avg_ips": self._safe_mean(cluster_data, "nb_ip")
        }
        
        return analysis
    
    def _get_top_values(self, df: pd.DataFrame, column: str, n: int = 3) -> Dict[str, int]:
        """Get top n values from a column as dictionary."""
        if column not in df.columns or df[column].empty:
            return {"N/A": 0}
        
        # Handle missing values
        valid_data = df[df[column].notna()]
        if valid_data.empty:
            return {"N/A": 0}
        
        return valid_data[column].value_counts().head(n).to_dict()
    
    def _safe_median(self, df: pd.DataFrame, column: str) -> Any:
        """Safely calculate median, handling missing values."""
        if column not in df.columns:
            return "N/A"
        
        valid_data = df[df[column].notna()]
        if valid_data.empty:
            return "N/A"
        
        try:
            return round(valid_data[column].median(), 2)
        except:
            return "N/A"
    
    def _safe_mean(self, df: pd.DataFrame, column: str) -> Any:
        """Safely calculate mean, handling missing values."""
        if column not in df.columns:
            return "N/A"
        
        valid_data = df[df[column].notna()]
        if valid_data.empty:
            return "N/A"
        
        try:
            return round(valid_data[column].mean(), 2)
        except:
            return "N/A"
    
    def profile_enterprises(self, df: pd.DataFrame) -> None:
        """Profile all enterprise clusters."""
        print("=== PROFILAGE ENTREPRISES ===")
        
        if df.empty:
            print("Aucune donnée d'entreprise disponible")
            return
        
        if "cluster_esn" not in df.columns:
            print("Colonne cluster_esn non trouvée dans les données d'entreprise")
            return
        
        clusters = sorted(df["cluster_esn"].dropna().unique())
        
        if not clusters:
            print("Aucun cluster d'entreprise trouvé")
            return
        
        for cluster_id in clusters:
            analysis = self.analyze_enterprise_cluster(df, cluster_id)
            
            if "error" in analysis:
                print(f"\n{analysis['error']}")
                continue
            
            print(f"\nCluster {analysis['cluster_id']} — {analysis['size']} entreprises")
            print("Secteurs dominants:", analysis["dominant_sectors"])
            print("Taille médiane:", analysis["median_size"])
            print("Fréq incidents médian:", analysis["median_incident_freq"])
            print("Impact moyen:", analysis["avg_impact"])
            print("Indispo moyenne:", analysis["avg_downtime"])
            print("Diversité attaques (moy):", analysis["avg_attack_diversity"])
    
    def profile_users(self, df: pd.DataFrame) -> None:
        """Profile all user clusters."""
        print("\n=== PROFILAGE UTILISATEURS ===")
        
        if df.empty:
            print("Aucune donnée d'utilisateur disponible")
            return
        
        if "cluster_risque" not in df.columns:
            print("Colonne cluster_risque non trouvée dans les données d'utilisateur")
            return
        
        clusters = sorted(df["cluster_risque"].dropna().unique())
        
        if not clusters:
            print("Aucun cluster d'utilisateur trouvé")
            return
        
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
        if df.empty or cluster_col not in df.columns:
            return pd.DataFrame(columns=["cluster_id", "cluster_type", "size", "percentage"])
        
        valid_data = df[df[cluster_col].notna()]
        if valid_data.empty:
            return pd.DataFrame(columns=["cluster_id", "cluster_type", "size", "percentage"])
        
        clusters = sorted(valid_data[cluster_col].unique())
        summaries = []
        
        for cluster_id in clusters:
            cluster_data = valid_data[valid_data[cluster_col] == cluster_id]
            
            summary = {
                "cluster_id": cluster_id,
                "cluster_type": cluster_type,
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(valid_data) * 100
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
            if not enterprises.empty and "cluster_esn" in enterprises.columns:
                ent_summary = self.generate_cluster_summary(enterprises, "cluster_esn", "enterprise")
                if not ent_summary.empty:
                    print("\n=== Résumé des Clusters d'Entreprises ===")
                    print(ent_summary)
                else:
                    print("\n=== Aucun cluster d'entreprise valide trouvé ===")
            
            if not users.empty and "cluster_risque" in users.columns:
                user_summary = self.generate_cluster_summary(users, "cluster_risque", "user")
                if not user_summary.empty:
                    print("\n=== Résumé des Clusters d'Utilisateurs ===")
                    print(user_summary)
                else:
                    print("\n=== Aucun cluster d'utilisateur valide trouvé ===")
            
            if enterprises.empty and users.empty:
                print("\n=== AUCUNE DONNÉE DE SEGMENTATION DISPONIBLE ===")
                print("Assurez-vous que les fichiers de segmentation existent et contiennent des données.")
                
        except Exception as e:
            print(f"Erreur lors du profilage: {e}")


def main():
    """Main execution function."""
    profiler = SegmentProfiler()
    profiler.run_complete_profiling()


if __name__ == "__main__":
    main()