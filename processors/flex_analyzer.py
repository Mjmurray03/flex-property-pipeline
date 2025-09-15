"""
Flex Property Analysis Dashboard Generator
Creates reports and visualizations of flex property candidates
"""
import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
from pathlib import Path

class FlexPropertyAnalyzer:
    """Analyze and report on flex property candidates"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.output_dir = Path('data/reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis of flex properties"""
        
        print("Generating Flex Property Analysis Report...")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'by_municipality': {},
            'by_property_use': {},
            'top_candidates': [],
            'value_analysis': {},
            'recommendations': []
        }
        
        # Get all enriched properties
        pipeline = [
            {'$match': {'flex_score': {'$exists': True}}},
            {'$sort': {'flex_score': -1}}
        ]
        
        properties = list(self.db.db.enriched_properties.aggregate(pipeline))
        
        if not properties:
            print("No enriched properties found")
            return report
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(properties)
        
        # Summary statistics
        report['summary'] = {
            'total_properties_analyzed': len(df),
            'flex_candidates': len(df[df['flex_score'] >= 5]),
            'prime_flex': len(df[df['flex_score'] >= 8]),
            'average_flex_score': float(df['flex_score'].mean()),
            'total_market_value': float(df['market_value'].sum()),
            'average_market_value': float(df['market_value'].mean()),
            'total_acres': float(df['acres'].sum())
        }
        
        # Analysis by municipality
        muni_analysis = df.groupby('municipality').agg({
            'flex_score': ['mean', 'max', 'count'],
            'market_value': 'mean',
            'acres': 'sum'
        }).round(2)
        
        for muni in muni_analysis.index:
            report['by_municipality'][muni] = {
                'count': int(muni_analysis.loc[muni, ('flex_score', 'count')]),
                'avg_score': float(muni_analysis.loc[muni, ('flex_score', 'mean')]),
                'max_score': float(muni_analysis.loc[muni, ('flex_score', 'max')]),
                'avg_value': float(muni_analysis.loc[muni, ('market_value', 'mean')]),
                'total_acres': float(muni_analysis.loc[muni, ('acres', 'sum')])
            }
        
        # Analysis by property use
        use_analysis = df.groupby('property_use').agg({
            'flex_score': ['mean', 'count'],
            'market_value': 'mean'
        }).round(2)
        
        for use in use_analysis.index:
            report['by_property_use'][use] = {
                'count': int(use_analysis.loc[use, ('flex_score', 'count')]),
                'avg_score': float(use_analysis.loc[use, ('flex_score', 'mean')]),
                'avg_value': float(use_analysis.loc[use, ('market_value', 'mean')])
            }
        
        # Top 20 candidates
        top_candidates = df.nlargest(20, 'flex_score')[
            ['parcel_id', 'address', 'municipality', 'property_use', 
             'flex_score', 'market_value', 'acres', 'owner_name']
        ].to_dict('records')
        
        report['top_candidates'] = top_candidates
        
        # Value analysis
        high_value = df[df['market_value'] > 1000000]
        report['value_analysis'] = {
            'properties_over_1m': len(high_value),
            'avg_score_over_1m': float(high_value['flex_score'].mean()) if len(high_value) > 0 else 0,
            'value_ranges': {
                'under_500k': len(df[df['market_value'] < 500000]),
                '500k_to_1m': len(df[(df['market_value'] >= 500000) & (df['market_value'] < 1000000)]),
                '1m_to_5m': len(df[(df['market_value'] >= 1000000) & (df['market_value'] < 5000000)]),
                'over_5m': len(df[df['market_value'] >= 5000000])
            }
        }
        
        # Generate recommendations
        report['recommendations'] = self.generate_recommendations(df)
        
        # Save report
        report_file = self.output_dir / f"flex_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {report_file}")
        
        # Also save top candidates to CSV
        csv_file = self.output_dir / f"top_flex_candidates_{datetime.now().strftime('%Y%m%d')}.csv"
        df.nlargest(100, 'flex_score').to_csv(csv_file, index=False)
        print(f"Top candidates CSV saved to: {csv_file}")
        
        return report
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Municipality recommendations
        top_muni = df.groupby('municipality')['flex_score'].mean().nlargest(3)
        if len(top_muni) > 0:
            recommendations.append(
                f"Focus on {', '.join(top_muni.index.tolist())} - highest average flex scores"
            )
        
        # Property use recommendations
        top_use = df.groupby('property_use')['flex_score'].mean().nlargest(2)
        if len(top_use) > 0:
            recommendations.append(
                f"Target {top_use.index[0]} properties - best flex potential"
            )
        
        # Value recommendations
        sweet_spot = df[(df['market_value'] >= 500000) & (df['market_value'] <= 2000000)]
        if len(sweet_spot) > 0:
            avg_score = sweet_spot['flex_score'].mean()
            recommendations.append(
                f"Sweet spot: Properties $500K-$2M have {avg_score:.1f} avg flex score"
            )
        
        # Size recommendations
        ideal_size = df[(df['acres'] >= 1) & (df['acres'] <= 5)]
        if len(ideal_size) > 0:
            recommendations.append(
                f"{len(ideal_size)} properties between 1-5 acres - ideal flex size"
            )
        
        # Opportunity recommendations
        high_score_low_value = df[(df['flex_score'] >= 7) & (df['market_value'] < 1000000)]
        if len(high_score_low_value) > 0:
            recommendations.append(
                f"Value opportunity: {len(high_score_low_value)} high-scoring properties under $1M"
            )
        
        return recommendations
    
    def generate_html_report(self, report: Dict) -> str:
        """Generate HTML dashboard from report data"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flex Property Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .card {{ background: white; padding: 15px; border-radius: 5px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .label {{ color: #666; font-size: 12px; }}
                table {{ width: 100%; border-collapse: collapse; background: white; }}
                th {{ background: #34495e; color: white; padding: 10px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background: #f5f5f5; }}
                .recommendations {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
                .score-high {{ color: #27ae60; font-weight: bold; }}
                .score-medium {{ color: #f39c12; font-weight: bold; }}
                .score-low {{ color: #e74c3c; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏭 Flex Property Analysis Dashboard</h1>
                <p>Palm Beach County - Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            
            <div class="summary">
                <div class="card">
                    <div class="label">Total Properties</div>
                    <div class="metric">{report['summary']['total_properties_analyzed']:,}</div>
                </div>
                <div class="card">
                    <div class="label">Flex Candidates</div>
                    <div class="metric">{report['summary']['flex_candidates']:,}</div>
                </div>
                <div class="card">
                    <div class="label">Prime Flex (Score 8+)</div>
                    <div class="metric">{report['summary']['prime_flex']:,}</div>
                </div>
                <div class="card">
                    <div class="label">Avg Flex Score</div>
                    <div class="metric">{report['summary']['average_flex_score']:.1f}</div>
                </div>
            </div>
            
            <h2>📊 Top Flex Property Candidates</h2>
            <table>
                <tr>
                    <th>Score</th>
                    <th>Address</th>
                    <th>Municipality</th>
                    <th>Type</th>
                    <th>Market Value</th>
                    <th>Acres</th>
                    <th>Owner</th>
                </tr>
        """
        
        for prop in report['top_candidates'][:20]:
            score = prop['flex_score']
            score_class = 'score-high' if score >= 8 else 'score-medium' if score >= 6 else 'score-low'
            
            html += f"""
                <tr>
                    <td class="{score_class}">{score:.1f}</td>
                    <td>{prop.get('address', 'N/A')}</td>
                    <td>{prop.get('municipality', 'N/A')}</td>
                    <td>{prop.get('property_use', 'N/A')}</td>
                    <td>${prop.get('market_value', 0):,.0f}</td>
                    <td>{prop.get('acres', 0):.2f}</td>
                    <td>{prop.get('owner_name', 'N/A')[:30]}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <div class="recommendations">
                <h3>💡 Key Recommendations</h3>
                <ul>
        """
        
        for rec in report['recommendations']:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
            
            <h2>📍 Analysis by Municipality</h2>
            <table>
                <tr>
                    <th>Municipality</th>
                    <th>Properties</th>
                    <th>Avg Score</th>
                    <th>Max Score</th>
                    <th>Avg Value</th>
                </tr>
        """
        
        for muni, data in sorted(report['by_municipality'].items(), 
                                key=lambda x: x[1]['avg_score'], reverse=True)[:10]:
            html += f"""
                <tr>
                    <td>{muni}</td>
                    <td>{data['count']}</td>
                    <td>{data['avg_score']:.1f}</td>
                    <td>{data['max_score']:.1f}</td>
                    <td>${data['avg_value']:,.0f}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # Save HTML report
        html_file = self.output_dir / f"flex_dashboard_{datetime.now().strftime('%Y%m%d')}.html"
        with open(html_file, 'w') as f:
            f.write(html)
        
        print(f"HTML dashboard saved to: {html_file}")
        
        return str(html_file)

# Standalone function for testing
def generate_report(db_manager):
    """Generate analysis report"""
    analyzer = FlexPropertyAnalyzer(db_manager)
    report = analyzer.generate_analysis_report()
    
    # Generate HTML dashboard
    if report['summary'].get('total_properties_analyzed', 0) > 0:
        html_file = analyzer.generate_html_report(report)
        print(f"\n✅ Analysis complete!")
        print(f"📊 View dashboard: {html_file}")
    
    return report

if __name__ == "__main__":
    # Test with sample data
    print("Run this after phase 2 enrichment to generate reports")