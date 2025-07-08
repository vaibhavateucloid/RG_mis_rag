# Python Comprehensive Excel-to-LLM Extractor with Values & Relationships
# Extracts actual data, calculated values, formulas, and relationships

import json
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string
import re
import os
from pathlib import Path
import datetime

class ComprehensiveExcelExtractor:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.workbook_values = None    # For calculated values
        self.workbook_formulas = None  # For formulas
        self.extraction = {
            'metadata': {},
            'business_context': {},
            'actual_data': {},
            'relationships': {},
            'data_flows': {},
            'key_metrics': {},
            'llm_summary': {}
        }
    
    def extract(self):
        """Main extraction method"""
        print("🔍 Starting comprehensive value + relationship extraction...")
        
        try:
            # Load workbook twice - once for values, once for formulas
            print("📚 Loading workbook for calculated values...")
            self.workbook_values = load_workbook(self.file_path, data_only=True)
            
            print("📚 Loading workbook for formulas...")
            self.workbook_formulas = load_workbook(self.file_path, data_only=False)
            
            # Extract all components
            self.extract_metadata()
            self.extract_business_context()
            self.extract_actual_data_with_context()
            self.extract_value_relationships()
            self.extract_business_flows()
            self.identify_key_metrics()
            self.create_llm_optimized_output()
            
            print("✅ Comprehensive extraction completed!")
            return self.extraction
            
        except Exception as error:
            print(f"❌ Extraction failed: {error}")
            raise error
    
    def extract_metadata(self):
        """Extract file metadata and assess data quality"""
        print("📋 Extracting metadata with business context...")
        
        sheet_names = [ws.title for ws in self.workbook_values.worksheets]
        self.extraction['metadata'] = {
            'filename': self.file_path.name,
            'total_sheets': len(self.workbook_values.worksheets),
            'sheet_names': sheet_names,
            'extraction_date': datetime.datetime.now().isoformat(),
            'business_type': 'Financial MIS System',
            'sheet_categories': self.categorize_sheets(sheet_names),
            'data_quality': self.assess_data_quality()
        }
    
    def categorize_sheets(self, sheet_names):
        """Categorize sheets by business function"""
        categories = {
            'executive_dashboards': [],
            'pnl_reporting': [],
            'allocation_engines': [],
            'source_data': [],
            'historical_analysis': [],
            'calculations': []
        }
        
        for sheet_name in sheet_names:
            lower = sheet_name.lower()
            if 'dashboard' in lower or 'console' in lower:
                categories['executive_dashboards'].append(sheet_name)
            elif 'p&l' in lower or 'profit' in lower:
                categories['pnl_reporting'].append(sheet_name)
            elif 'allocation' in lower:
                categories['allocation_engines'].append(sheet_name)
            elif 'daas' in lower or 'format' in lower:
                categories['source_data'].append(sheet_name)
            elif 'ly' in lower or 'last year' in lower:
                categories['historical_analysis'].append(sheet_name)
            else:
                categories['calculations'].append(sheet_name)
        
        return categories
    
    def assess_data_quality(self):
        """Assess overall data quality"""
        total_cells = 0
        populated_cells = 0
        error_cells = 0
        
        # Sample first 5 sheets for quality assessment
        for sheet in self.workbook_values.worksheets[:5]:
            for row in sheet.iter_rows(max_row=50, max_col=20):
                for cell in row:
                    total_cells += 1
                    if cell.value is not None and cell.value != '':
                        populated_cells += 1
                        if isinstance(cell.value, str) and any(err in str(cell.value) for err in ['#REF!', '#ERROR', '#VALUE!']):
                            error_cells += 1
        
        return {
            'population_rate': round((populated_cells / total_cells) * 100) if total_cells > 0 else 0,
            'error_rate': round((error_cells / populated_cells) * 100) if populated_cells > 0 else 0,
            'quality_score': round(((populated_cells - error_cells) / populated_cells) * 100) if populated_cells > 0 else 0
        }
    
    def extract_business_context(self):
        """Extract business context from sheet names and structure"""
        print("💼 Extracting business context...")
        
        business_units = set()
        geographies = set()
        time_horizons = set()
        product_lines = set()
        
        for name in self.extraction['metadata']['sheet_names']:
            lower = name.lower()
            
            # Business units
            if 'console' in lower: business_units.add('Console')
            if 'hospi' in lower: business_units.add('Hospitality BI')
            if 'rezgain' in lower: business_units.add('RezGain')
            if 'dhisco' in lower: business_units.add('DHISCO')
            if 'bcv' in lower: business_units.add('BCV')
            
            # Geographies
            if 'apmea' in lower: geographies.add('APMEA')
            if 'latam' in lower: geographies.add('LATAM')
            
            # Time periods
            if "fy'25" in lower or '25' in lower: time_horizons.add('FY25')
            if "fy'24" in lower or '24' in lower: time_horizons.add('FY24')
            if "fy'23" in lower or '23' in lower: time_horizons.add('FY23')
            
            # Product lines
            if 'ota' in lower: product_lines.add('OTA')
            if 'car' in lower: product_lines.add('Car Rental')
            if 'air' in lower: product_lines.add('Air Travel')
            if 'cruise' in lower: product_lines.add('Cruise')
        
        self.extraction['business_context'] = {
            'business_units': list(business_units),
            'geographies': list(geographies),
            'time_horizons': list(time_horizons),
            'product_lines': list(product_lines),
            'business_model': 'Multi-BU Travel Technology Platform',
            'reporting_structure': 'Matrix organization with geographic and product dimensions'
        }
    
    def extract_actual_data_with_context(self):
        """Extract actual data with business context"""
        print("📊 Extracting actual data with business context...")
        
        # Focus on key business sheets
        key_sheets = [
            'Console Charts', 'CEO Dashboard', 'Summary', 'New Summary',
            'BU Wise PL FY\'25', '>> Profit & Loss', 'Cash Flow'
        ]
        
        for sheet_name in key_sheets:
            if sheet_name in [ws.title for ws in self.workbook_values.worksheets]:
                print(f"  Processing {sheet_name}...")
                self.extraction['actual_data'][sheet_name] = self.extract_sheet_data_with_values(sheet_name)
    
    def extract_sheet_data_with_values(self, sheet_name):
        """Extract comprehensive data from a single sheet"""
        try:
            # Get both versions of the sheet
            value_sheet = self.workbook_values[sheet_name]
            formula_sheet = self.workbook_formulas[sheet_name] if sheet_name in [ws.title for ws in self.workbook_formulas.worksheets] else None
            
            # Convert to pandas for easier handling
            data_values = []
            max_row = min(value_sheet.max_row, 200)  # Limit to first 200 rows
            max_col = min(value_sheet.max_column, 50)  # Limit to first 50 columns
            
            for row in value_sheet.iter_rows(min_row=1, max_row=max_row, max_col=max_col, values_only=True):
                data_values.append(list(row))
            
            # Create DataFrame
            df = pd.DataFrame(data_values)
            df = df.replace({None: np.nan})
            
            # Extract cell-by-cell details
            cell_details = {}
            business_metrics = []
            
            for row_idx in range(1, min(max_row + 1, 101)):  # First 100 rows
                for col_idx in range(1, min(max_col + 1, 31)):  # First 30 columns
                    value_cell = value_sheet.cell(row_idx, col_idx)
                    formula_cell = formula_sheet.cell(row_idx, col_idx) if formula_sheet else None
                    
                    if value_cell.value is not None or (formula_cell and formula_cell.value is not None):
                        cell_addr = f"{get_column_letter(col_idx)}{row_idx}"
                        
                        cell_info = {
                            'address': cell_addr,
                            'row': row_idx,
                            'column': get_column_letter(col_idx),
                            'value': value_cell.value,
                            'display_value': self.get_display_value(value_cell),
                            'formula': formula_cell.value if formula_cell and isinstance(formula_cell.value, str) and formula_cell.value.startswith('=') else None,
                            'data_type': type(value_cell.value).__name__,
                            'business_context': self.infer_cell_business_context(sheet_name, row_idx, col_idx, value_cell, df)
                        }
                        
                        cell_details[cell_addr] = cell_info
                        
                        # Identify business metrics
                        if self.is_business_metric(cell_info, df, row_idx - 1, col_idx - 1):
                            business_metrics.append(cell_info)
            
            return {
                'sheet_name': sheet_name,
                'structured_data': df.fillna(None).values.tolist(),
                'column_headers': self.extract_column_headers(df),
                'row_headers': self.extract_row_headers(df),
                'cell_details': cell_details,
                'business_metrics': business_metrics,
                'summary': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'metrics_count': len(business_metrics),
                    'has_formulas': any(cell.get('formula') for cell in cell_details.values()),
                    'data_quality': self.assess_sheet_data_quality(cell_details)
                }
            }
            
        except Exception as e:
            print(f"  ⚠️ Error processing {sheet_name}: {e}")
            return {'error': str(e)}
    
    def get_display_value(self, cell):
        """Get the display value of a cell"""
        if cell.value is None:
            return None
        elif isinstance(cell.value, (int, float)):
            return cell.value
        else:
            return str(cell.value)
    
    def infer_cell_business_context(self, sheet_name, row, col, cell, df):
        """Infer business context of a cell"""
        if cell.value is None:
            return 'empty'
        
        value = cell.value
        
        # Check if it's a label/header
        if isinstance(value, str):
            lower_value = value.lower()
            if 'revenue' in lower_value: return 'revenue_metric'
            elif 'cost' in lower_value or 'expense' in lower_value: return 'cost_metric'
            elif 'profit' in lower_value or 'margin' in lower_value: return 'profitability_metric'
            elif 'allocation' in lower_value: return 'allocation_driver'
            elif 'booking' in lower_value: return 'booking_metric'
            return 'label'
        
        # Check if it's a numeric business value
        if isinstance(value, (int, float)):
            # Get context from headers
            row_header = self.get_row_header(df, row - 1)
            col_header = self.get_column_header(df, col - 1)
            
            if row_header and col_header:
                row_lower = str(row_header).lower()
                col_lower = str(col_header).lower()
                
                if 'revenue' in row_lower and abs(value) > 1000: return 'revenue_value'
                elif 'cost' in row_lower and abs(value) > 1000: return 'cost_value'
                elif 'margin' in row_lower and -100 <= value <= 100: return 'margin_percentage'
                elif 'act' in col_lower or 'actual' in col_lower: return 'actual_performance'
                elif 'bud' in col_lower or 'budget' in col_lower: return 'budget_target'
            
            return 'numeric_value'
        
        return 'other'
    
    def get_row_header(self, df, row_idx):
        """Get row header (usually first column)"""
        if row_idx < len(df) and len(df.columns) > 0:
            value = df.iloc[row_idx, 0]
            return value if pd.notna(value) else None
        return None
    
    def get_column_header(self, df, col_idx):
        """Get column header (look in first few rows)"""
        if col_idx < len(df.columns):
            for row_idx in range(min(5, len(df))):
                value = df.iloc[row_idx, col_idx]
                if pd.notna(value) and isinstance(value, str) and len(str(value)) < 50:
                    return value
        return None
    
    def extract_column_headers(self, df):
        """Extract column headers from DataFrame"""
        headers = []
        for col_idx in range(len(df.columns)):
            header = self.get_column_header(df, col_idx)
            headers.append(header)
        return headers
    
    def extract_row_headers(self, df):
        """Extract row headers from DataFrame"""
        headers = []
        for row_idx in range(len(df)):
            header = self.get_row_header(df, row_idx)
            headers.append(header)
        return headers
    
    def is_business_metric(self, cell_info, df, row_idx, col_idx):
        """Determine if a cell contains a business metric"""
        if cell_info['value'] is None or not isinstance(cell_info['value'], (int, float)):
            return False
        if abs(cell_info['value']) < 1:  # Too small to be significant
            return False
        
        row_header = self.get_row_header(df, row_idx)
        if not row_header:
            return False
        
        business_terms = [
            'revenue', 'cost', 'expense', 'profit', 'margin', 'ebitda',
            'booking', 'growth', 'allocation', 'total', 'budget', 'actual'
        ]
        
        return any(term in str(row_header).lower() for term in business_terms)
    
    def assess_sheet_data_quality(self, cell_details):
        """Assess data quality for a specific sheet"""
        total_cells = len(cell_details)
        numeric_cells = sum(1 for cell in cell_details.values() if isinstance(cell['value'], (int, float)))
        error_cells = sum(1 for cell in cell_details.values() 
                         if isinstance(cell['value'], str) and '#' in str(cell['value']))
        
        return {
            'total_cells': total_cells,
            'numeric_percentage': round((numeric_cells / total_cells) * 100) if total_cells > 0 else 0,
            'error_percentage': round((error_cells / total_cells) * 100) if total_cells > 0 else 0
        }
    
    def extract_value_relationships(self):
        """Extract relationships between values"""
        print("🔗 Extracting value relationships...")
        
        self.extraction['relationships'] = {}
        
        for sheet_name, sheet_data in self.extraction['actual_data'].items():
            if 'error' in sheet_data:
                continue
                
            relationships = []
            
            # Find cross-sheet references with actual values
            for cell_addr, cell_info in sheet_data['cell_details'].items():
                if cell_info['formula'] and '!' in cell_info['formula']:
                    refs = self.parse_formula_references(cell_info['formula'])
                    for ref in refs:
                        relationships.append({
                            'source_cell': cell_addr,
                            'source_value': cell_info['value'],
                            'target_sheet': ref['sheet'],
                            'target_range': ref['range'],
                            'formula': cell_info['formula'],
                            'business_context': cell_info['business_context'],
                            'relationship_type': self.classify_relationship(cell_info['formula'], cell_info['business_context'])
                        })
            
            self.extraction['relationships'][sheet_name] = {
                'cross_sheet_references': relationships,
                'business_logic': self.infer_business_logic(relationships, sheet_data)
            }
    
    def parse_formula_references(self, formula):
        """Parse cross-sheet references from formulas"""
        refs = []
        pattern = r"(?:'([^']+)'|([^!']+))!([A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?)"
        matches = re.findall(pattern, formula)
        
        for match in matches:
            sheet_name = match[0] or match[1]
            cell_range = match[2]
            refs.append({'sheet': sheet_name, 'range': cell_range})
        
        return refs
    
    def classify_relationship(self, formula, business_context):
        """Classify the type of relationship"""
        formula_upper = formula.upper()
        
        if 'SUM(' in formula_upper:
            return 'aggregation'
        elif '*' in formula and 'allocation' in business_context:
            return 'allocation_calculation'
        elif 'VLOOKUP' in formula_upper or 'INDEX' in formula_upper:
            return 'data_lookup'
        elif 'IF(' in formula_upper:
            return 'conditional_logic'
        elif 'revenue' in business_context or 'cost' in business_context:
            return 'financial_calculation'
        else:
            return 'reference'
    
    def infer_business_logic(self, relationships, sheet_data):
        """Infer business logic from relationships"""
        sheet_name = sheet_data['sheet_name'].lower()
        
        if 'dashboard' in sheet_name or 'console' in sheet_name:
            return {
                'primary_function': 'executive_reporting',
                'data_flow': 'aggregation_from_multiple_sources',
                'complexity': 'high' if len(relationships) > 20 else 'medium'
            }
        elif 'allocation' in sheet_name:
            return {
                'primary_function': 'cost_allocation',
                'data_flow': 'distribution_to_business_units',
                'complexity': 'high' if len(relationships) > 10 else 'medium'
            }
        else:
            return {
                'primary_function': 'calculation',
                'data_flow': 'processing',
                'complexity': 'low' if len(relationships) < 5 else 'medium'
            }
    
    def extract_business_flows(self):
        """Extract business process flows"""
        print("💼 Extracting business flows...")
        
        self.extraction['data_flows'] = {
            'revenue_flow': self.trace_business_flow('revenue'),
            'cost_flow': self.trace_business_flow('cost'),
            'allocation_flow': self.trace_business_flow('allocation'),
            'reporting_flow': self.trace_business_flow('dashboard')
        }
    
    def trace_business_flow(self, flow_type):
        """Trace a specific business flow"""
        relevant_sheets = []
        
        for sheet_name in self.extraction['actual_data'].keys():
            lower_name = sheet_name.lower()
            if (flow_type in lower_name or 
                (flow_type == 'dashboard' and ('console' in lower_name or 'dashboard' in lower_name))):
                relevant_sheets.append(sheet_name)
        
        return {
            'sheets': relevant_sheets,
            'flow_pattern': 'identified' if relevant_sheets else 'not_found'
        }
    
    def identify_key_metrics(self):
        """Identify key business metrics"""
        print("🎯 Identifying key business metrics...")
        
        all_metrics = []
        
        for sheet_name, sheet_data in self.extraction['actual_data'].items():
            if 'business_metrics' in sheet_data:
                for metric in sheet_data['business_metrics']:
                    all_metrics.append({
                        **metric,
                        'sheet': sheet_name,
                        'importance': self.calculate_metric_importance(metric, sheet_data)
                    })
        
        # Sort by importance
        top_metrics = sorted(all_metrics, key=lambda x: x['importance'], reverse=True)[:50]
        
        self.extraction['key_metrics'] = {
            'total_identified': len(all_metrics),
            'top_metrics': top_metrics,
            'by_category': self.categorize_metrics(top_metrics)
        }
    
    def calculate_metric_importance(self, metric, sheet_data):
        """Calculate importance score for a metric"""
        importance = 0
        
        # Higher importance for dashboard sheets
        if 'dashboard' in sheet_data['sheet_name'].lower():
            importance += 10
        
        # Higher importance for revenue/profit metrics
        if 'revenue' in metric['business_context'] or 'profit' in metric['business_context']:
            importance += 8
        
        # Higher importance for larger values
        if isinstance(metric['value'], (int, float)):
            abs_value = abs(metric['value'])
            if abs_value > 1000000:
                importance += 5
            elif abs_value > 100000:
                importance += 3
            elif abs_value > 10000:
                importance += 1
        
        # Higher importance if it has a formula
        if metric['formula']:
            importance += 2
        
        return importance
    
    def categorize_metrics(self, metrics):
        """Categorize metrics by type"""
        categories = {
            'revenue': [],
            'costs': [],
            'profitability': [],
            'performance': [],
            'other': []
        }
        
        for metric in metrics:
            context = metric['business_context'].lower()
            if 'revenue' in context:
                categories['revenue'].append(metric)
            elif 'cost' in context:
                categories['costs'].append(metric)
            elif 'profit' in context or 'margin' in context:
                categories['profitability'].append(metric)
            elif 'actual' in context or 'budget' in context:
                categories['performance'].append(metric)
            else:
                categories['other'].append(metric)
        
        return categories
    
    def create_llm_optimized_output(self):
        """Create LLM-optimized summary"""
        print("🤖 Creating LLM-optimized output...")
        
        self.extraction['llm_summary'] = {
            'executive_summary': {
                'business_type': self.extraction['business_context']['business_model'],
                'total_sheets': self.extraction['metadata']['total_sheets'],
                'key_business_units': self.extraction['business_context']['business_units'],
                'data_quality_score': self.extraction['metadata']['data_quality']['quality_score'],
                'top_metrics_count': len(self.extraction['key_metrics']['top_metrics']) if 'key_metrics' in self.extraction else 0
            },
            'business_structure': {
                'organizational_dimensions': self.extraction['business_context'],
                'reporting_hierarchy': self.extraction['metadata']['sheet_categories']
            },
            'financial_metrics': self.extraction['key_metrics']['by_category'] if 'key_metrics' in self.extraction else {},
            'sample_data': self.create_sample_data_for_llm()
        }
    
    def create_sample_data_for_llm(self):
        """Create sample data for LLM analysis"""
        samples = {}
        
        for sheet_name, sheet_data in self.extraction['actual_data'].items():
            if 'error' not in sheet_data:
                samples[sheet_name] = {
                    'purpose': 'Business metrics' if sheet_data['business_metrics'] else 'Supporting calculations',
                    'sample_rows': sheet_data['structured_data'][:10],  # First 10 rows
                    'column_headers': sheet_data['column_headers'][:10],  # First 10 columns
                    'key_metrics': sheet_data['business_metrics'][:5],  # Top 5 metrics
                    'summary': sheet_data['summary']
                }
        
        return samples
    
    def save_llm_ready_data(self, output_dir='./llm_ready_data'):
        """Save LLM-ready data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete LLM-optimized data
        with open(f"{output_dir}/complete_business_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.extraction['llm_summary'], f, indent=2, default=str)
        
        # Save focused datasets
        with open(f"{output_dir}/financial_metrics.json", 'w', encoding='utf-8') as f:
            json.dump({
                'overview': self.extraction['llm_summary']['executive_summary'],
                'metrics': self.extraction['llm_summary']['financial_metrics'],
                'relationships': {k: v['cross_sheet_references'][:10] for k, v in self.extraction['relationships'].items()}
            }, f, indent=2, default=str)
        
        with open(f"{output_dir}/sample_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.extraction['llm_summary']['sample_data'], f, indent=2, default=str)
        
        # Create analysis prompt
        prompt = self.create_llm_prompt()
        with open(f"{output_dir}/llm_analysis_prompt.md", 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"✅ LLM-ready data saved to: {output_dir}")
        return output_dir
    
    def create_llm_prompt(self):
        """Create LLM analysis prompt"""
        return f"""# Financial MIS Analysis Request

## Business Context
{json.dumps(self.extraction['llm_summary']['executive_summary'], indent=2)}

## Key Analysis Questions:

1. **Business Structure Analysis**
   - How are the {len(self.extraction['business_context']['business_units'])} business units structured?
   - What are the key revenue and cost drivers?
   - How do geographic allocations work?

2. **Financial Flow Analysis**
   - How does revenue flow through the system?
   - What allocation methodologies are used?
   - How are variances calculated?

3. **Process Optimization**
   - Where are the bottlenecks?
   - What can be simplified?
   - What are the control points?

## Sample Data Structure
{json.dumps(list(self.extraction['llm_summary']['sample_data'].keys()), indent=2)}

## Key Metrics Categories
{json.dumps({k: len(v) for k, v in self.extraction['llm_summary']['financial_metrics'].items()}, indent=2)}

Please analyze this financial MIS system focusing on business relationships, data flows, and optimization opportunities.
"""

# Usage
def main():
    extractor = ComprehensiveExcelExtractor('./MIS Master File FY25.xlsx')
    
    try:
        print("🚀 Starting comprehensive Excel-to-LLM extraction...")
        
        extraction = extractor.extract()
        output_dir = extractor.save_llm_ready_data()
        
        print("\n🎉 Extraction Complete!")
        print(f"\n📊 Results:")
        print(f"- Business Units: {', '.join(extraction['business_context']['business_units'])}")
        print(f"- Data Quality: {extraction['metadata']['data_quality']['quality_score']}%")
        print(f"- Key Metrics: {extraction['llm_summary']['executive_summary']['top_metrics_count']}")
        print(f"\n📁 Files created in: {output_dir}")
        
    except Exception as error:
        print(f"❌ Error: {error}")

if __name__ == '__main__':
    main()