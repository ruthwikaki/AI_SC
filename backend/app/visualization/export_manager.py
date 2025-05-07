"""
Export manager module for the supply chain LLM platform.

This module provides functionality to export visualizations and data
in various formats for external use.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
import pandas as pd
import numpy as np
import json
import io
import base64
import csv
import zipfile
from datetime import datetime
import re
import os
import matplotlib.pyplot as plt

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ExportManager:
    """
    Handles export of visualizations and data.
    
    This class provides methods to export charts, data, and analytics
    in various formats such as PNG, CSV, Excel, PDF, etc.
    """
    
    def __init__(self):
        """Initialize the export manager."""
        self.supported_formats = {
            "image": ["png", "jpg", "svg"],
            "data": ["csv", "xlsx", "json"],
            "document": ["pdf", "html"]
        }
        
    def export_chart(
        self,
        chart_data: Dict[str, Any],
        format: str = "png",
        include_data: bool = False
    ) -> Dict[str, Any]:
        """
        Export a chart in the specified format.
        
        Args:
            chart_data: Chart data including image and metadata
            format: Export format ('png', 'jpg', 'svg', 'pdf', etc.)
            include_data: Whether to include underlying data
            
        Returns:
            Dictionary with export data
        """
        try:
            format = format.lower()
            
            # Check if format is supported
            if (format not in self.supported_formats["image"] and 
                format not in self.supported_formats["document"]):
                raise ValueError(f"Unsupported export format: {format}")
                
            # Extract image data
            image_data = chart_data.get("image", "")
            
            # Check if image data is available
            if not image_data:
                raise ValueError("No image data available in chart data")
                
            # Process based on format
            if format in self.supported_formats["image"]:
                # Export as image
                export_data = self._export_as_image(image_data, format)
            elif format in self.supported_formats["document"]:
                # Export as document
                export_data = self._export_as_document(chart_data, format)
            
            # Include original data if requested
            if include_data:
                if "data" in chart_data:
                    # If data is given as a list of dictionaries
                    raw_data = chart_data["data"]
                    
                    # Export data as CSV
                    if isinstance(raw_data, list):
                        df = pd.DataFrame(raw_data)
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        export_data["data_csv"] = csv_data
                        
                        # Also include JSON for convenience
                        export_data["data_json"] = json.dumps(raw_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting chart: {str(e)}")
            raise
            
    def export_data(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        format: str = "csv",
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export data in the specified format.
        
        Args:
            data: Data to export
            format: Export format ('csv', 'xlsx', 'json')
            filename: Optional filename without extension
            
        Returns:
            Dictionary with export data
        """
        try:
            format = format.lower()
            
            # Check if format is supported
            if format not in self.supported_formats["data"]:
                raise ValueError(f"Unsupported data export format: {format}")
                
            # Convert to DataFrame if list or dict
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    # Extract data from chart data
                    df = pd.DataFrame(data["data"])
                else:
                    # Try to convert dictionary to DataFrame
                    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
                    df.columns = ["Key", "Value"]
            else:
                df = data.copy()
                
            # Generate default filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"export_{timestamp}"
                
            # Generate clean filename
            clean_filename = self._clean_filename(filename)
                
            # Export based on format
            if format == "csv":
                buffer = io.StringIO()
                df.to_csv(buffer, index=False)
                export_data = {
                    "format": "csv",
                    "filename": f"{clean_filename}.csv",
                    "mime_type": "text/csv",
                    "data": buffer.getvalue()
                }
            elif format == "xlsx":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="Data", index=False)
                    
                    # Get workbook and worksheet
                    workbook = writer.book
                    worksheet = writer.sheets["Data"]
                    
                    # Add filters and autofit columns
                    worksheet.autofilter(0, 0, 0, len(df.columns) - 1)
                    for i, col in enumerate(df.columns):
                        col_len = max(df[col].astype(str).map(len).max(), len(str(col)))
                        worksheet.set_column(i, i, col_len + 2)
                
                export_data = {
                    "format": "xlsx",
                    "filename": f"{clean_filename}.xlsx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "data": base64.b64encode(buffer.getvalue()).decode("utf-8")
                }
            elif format == "json":
                if isinstance(data, pd.DataFrame):
                    # Convert DataFrame to list of dictionaries
                    json_data = data.to_dict(orient="records")
                else:
                    json_data = data
                
                export_data = {
                    "format": "json",
                    "filename": f"{clean_filename}.json",
                    "mime_type": "application/json",
                    "data": json.dumps(json_data, default=self._json_serializer)
                }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def export_multiple(
        self, 
        items: List[Dict[str, Any]], 
        format: str = "zip"
    ) -> Dict[str, Any]:
        """
        Export multiple items (charts, data) into a single archive.
        
        Args:
            items: List of items to export
            format: Export format ('zip')
            
        Returns:
            Dictionary with export data
        """
        try:
            if format.lower() != "zip":
                raise ValueError(f"Unsupported multiple export format: {format}")
                
            buffer = io.BytesIO()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, item in enumerate(items):
                    item_type = item.get("type", "data")
                    item_data = item.get("data")
                    item_format = item.get("format", "csv" if item_type == "data" else "png")
                    item_filename = item.get("filename", f"{item_type}_{i+1}")
                    
                    # Clean filename
                    clean_filename = self._clean_filename(item_filename)
                    
                    if item_type == "chart":
                        # Export chart
                        chart_export = self.export_chart(
                            item_data, 
                            format=item_format, 
                            include_data=item.get("include_data", False)
                        )
                        
                        # Add chart to zip
                        zf.writestr(
                            f"{clean_filename}.{item_format}", 
                            base64.b64decode(chart_export["data"]) if "data" in chart_export else chart_export["data_raw"]
                        )
                        
                        # Add data if included
                        if "data_csv" in chart_export:
                            zf.writestr(f"{clean_filename}_data.csv", chart_export["data_csv"])
                    
                    elif item_type == "data":
                        # Export data
                        data_export = self.export_data(
                            item_data, 
                            format=item_format, 
                            filename=clean_filename
                        )
                        
                        # Handle binary vs text data
                        if item_format == "xlsx":
                            zf.writestr(
                                data_export["filename"], 
                                base64.b64decode(data_export["data"])
                            )
                        else:
                            zf.writestr(
                                data_export["filename"], 
                                data_export["data"]
                            )
            
            export_data = {
                "format": "zip",
                "filename": f"export_archive_{timestamp}.zip",
                "mime_type": "application/zip",
                "data": base64.b64encode(buffer.getvalue()).decode("utf-8")
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting multiple items: {str(e)}")
            raise
            
    def export_report(
        self, 
        report_data: Dict[str, Any], 
        format: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Export a report with multiple sections, charts, and tables.
        
        Args:
            report_data: Report content and structure
            format: Export format ('pdf', 'html')
            
        Returns:
            Dictionary with export data
        """
        try:
            format = format.lower()
            
            if format not in self.supported_formats["document"]:
                raise ValueError(f"Unsupported report format: {format}")
                
            # Generate default filename if not provided
            report_title = report_data.get("title", "Report")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self._clean_filename(report_title)}_{timestamp}"
            
            # Process based on format
            if format == "pdf":
                export_data = self._export_report_as_pdf(report_data, filename)
            elif format == "html":
                export_data = self._export_report_as_html(report_data, filename)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            raise
    
    def _export_as_image(self, image_data: str, format: str) -> Dict[str, Any]:
        """
        Export image data in the specified format.
        
        Args:
            image_data: Base64 encoded image data or raw image data
            format: Image format ('png', 'jpg', 'svg')
            
        Returns:
            Dictionary with export data
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if image_data is already base64 encoded
            if image_data.startswith("data:image"):
                # Extract base64 data
                image_base64 = image_data.split(",")[1]
            elif image_data.startswith("<svg"):
                # Handle SVG data
                if format == "svg":
                    # Return as is
                    return {
                        "format": "svg",
                        "filename": f"chart_{timestamp}.svg",
                        "mime_type": "image/svg+xml",
                        "data_raw": image_data,
                        "data": base64.b64encode(image_data.encode("utf-8")).decode("utf-8")
                    }
                else:
                    # Convert SVG to other format
                    return self._convert_svg_to_format(image_data, format, timestamp)
            else:
                # Assume it's already base64 encoded
                image_base64 = image_data
            
            # Determine mime type based on format
            mime_type = f"image/{format}"
            
            return {
                "format": format,
                "filename": f"chart_{timestamp}.{format}",
                "mime_type": mime_type,
                "data": image_base64
            }
            
        except Exception as e:
            logger.error(f"Error exporting image: {str(e)}")
            raise
    
    def _export_as_document(self, chart_data: Dict[str, Any], format: str) -> Dict[str, Any]:
        """
        Export chart as a document (PDF or HTML).
        
        Args:
            chart_data: Chart data including image and metadata
            format: Document format ('pdf', 'html')
            
        Returns:
            Dictionary with export data
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            title = chart_data.get("title", "Chart")
            description = chart_data.get("description", "")
            
            if format == "pdf":
                # Create PDF document with chart
                pdf_buffer = self._create_pdf_with_chart(chart_data)
                
                return {
                    "format": "pdf",
                    "filename": f"{self._clean_filename(title)}_{timestamp}.pdf",
                    "mime_type": "application/pdf",
                    "data": base64.b64encode(pdf_buffer.getvalue()).decode("utf-8")
                }
                
            elif format == "html":
                # Create HTML document with chart
                html_content = self._create_html_with_chart(chart_data)
                
                return {
                    "format": "html",
                    "filename": f"{self._clean_filename(title)}_{timestamp}.html",
                    "mime_type": "text/html",
                    "data": base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
                }
                
        except Exception as e:
            logger.error(f"Error exporting as document: {str(e)}")
            raise
    
    def _convert_svg_to_format(self, svg_data: str, format: str, timestamp: str) -> Dict[str, Any]:
        """
        Convert SVG to another image format.
        
        Args:
            svg_data: SVG content as string
            format: Target format ('png', 'jpg')
            timestamp: Timestamp for filename
            
        Returns:
            Dictionary with export data
        """
        try:
            # For this example, we'll use matplotlib to convert SVG to other formats
            # In a real implementation, you might want to use a more specialized library
            
            buffer = io.BytesIO()
            plt.figure(figsize=(10, 6))
            
            # Save SVG to a temp file
            temp_svg = io.StringIO()
            temp_svg.write(svg_data)
            temp_svg.seek(0)
            
            # Read the SVG into matplotlib
            img = plt.imread(temp_svg, format="svg")
            plt.imshow(img)
            plt.axis('off')
            
            # Save as requested format
            plt.savefig(buffer, format=format, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Create response
            return {
                "format": format,
                "filename": f"chart_{timestamp}.{format}",
                "mime_type": f"image/{format}",
                "data": base64.b64encode(buffer.getvalue()).decode("utf-8")
            }
            
        except Exception as e:
            logger.error(f"Error converting SVG to {format}: {str(e)}")
            # Fallback to returning original SVG
            return {
                "format": "svg",
                "filename": f"chart_{timestamp}.svg",
                "mime_type": "image/svg+xml",
                "data_raw": svg_data,
                "data": base64.b64encode(svg_data.encode("utf-8")).decode("utf-8")
            }
    
    def _create_pdf_with_chart(self, chart_data: Dict[str, Any]) -> io.BytesIO:
        """
        Create a PDF document containing the chart and its metadata.
        
        Args:
            chart_data: Chart data including image and metadata
            
        Returns:
            BytesIO buffer with PDF content
        """
        try:
            # In a real implementation, you would use a PDF library like ReportLab
            # For this example, we'll just create a simple PDF structure
            
            # This is a placeholder - in a real implementation you would:
            # 1. Create a PDF document with ReportLab or similar
            # 2. Add chart title, description, and metadata
            # 3. Add the chart image
            # 4. Add data tables if included
            
            buffer = io.BytesIO()
            
            # Placeholder for PDF creation logic
            # This would be where you'd use a PDF generation library
            
            # For now, we'll just create some dummy PDF content
            pdf_content = f"""
            %PDF-1.4
            1 0 obj
            << /Type /Catalog
               /Pages 2 0 R
            >>
            endobj
            2 0 obj
            << /Type /Pages
               /Kids [3 0 R]
               /Count 1
            >>
            endobj
            3 0 obj
            << /Type /Page
               /Parent 2 0 R
               /Resources << /Font << /F1 4 0 R >> >>
               /MediaBox [0 0 612 792]
               /Contents 5 0 R
            >>
            endobj
            4 0 obj
            << /Type /Font
               /Subtype /Type1
               /Name /F1
               /BaseFont /Helvetica
            >>
            endobj
            5 0 obj
            << /Length 67 >>
            stream
            BT
            /F1 24 Tf
            72 720 Td
            ({chart_data.get("title", "Chart")}) Tj
            ET
            endstream
            endobj
            xref
            0 6
            0000000000 65535 f
            0000000009 00000 n
            0000000058 00000 n
            0000000115 00000 n
            0000000234 00000 n
            0000000321 00000 n
            trailer
            << /Size 6
               /Root 1 0 R
            >>
            startxref
            439
            %%EOF
            """
            
            buffer.write(pdf_content.encode("utf-8"))
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            raise
    
    def _create_html_with_chart(self, chart_data: Dict[str, Any]) -> str:
        """
        Create an HTML document containing the chart and its metadata.
        
        Args:
            chart_data: Chart data including image and metadata
            
        Returns:
            HTML content as string
        """
        try:
            title = chart_data.get("title", "Chart")
            description = chart_data.get("description", "")
            image_data = chart_data.get("image", "")
            
            # Check if image_data is already in data URI format
            if not image_data.startswith("data:image"):
                # If it's SVG content
                if image_data.startswith("<svg"):
                    image_html = image_data
                else:
                    # Assume it's base64 encoded - determine format
                    if "image_format" in chart_data:
                        image_format = chart_data["image_format"]
                    else:
                        # Default to PNG
                        image_format = "png"
                    
                    image_html = f'<img src="data:image/{image_format};base64,{image_data}" alt="{title}" style="max-width:100%;">'
            else:
                # Already data URI
                image_html = f'<img src="{image_data}" alt="{title}" style="max-width:100%;">'
            
            # Create HTML template
            html_content = f"""<!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .chart-title {{ font-size: 24px; font-weight: bold; margin-bottom: 10px; }}
                    .chart-description {{ margin-bottom: 20px; color: #555; }}
                    .chart-container {{ margin-bottom: 30px; }}
                    .metadata {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                    .metadata-title {{ font-weight: bold; margin-bottom: 10px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="chart-title">{title}</h1>
                    <div class="chart-description">{description}</div>
                    
                    <div class="chart-container">
                        {image_html}
                    </div>
                    
                    <div class="metadata">
                        <div class="metadata-title">Chart Information</div>
                        <table>
                            <tr>
                                <th>Property</th>
                                <th>Value</th>
                            </tr>
            """
            
            # Add metadata
            for key, value in chart_data.items():
                # Skip image and data fields
                if key not in ["image", "data"]:
                    # Format value for display
                    if isinstance(value, dict) or isinstance(value, list):
                        display_value = json.dumps(value)
                    else:
                        display_value = str(value)
                    
                    html_content += f"""
                            <tr>
                                <td>{key}</td>
                                <td>{display_value}</td>
                            </tr>
                    """
            
            # Add data table if available
            if "data" in chart_data and isinstance(chart_data["data"], list) and len(chart_data["data"]) > 0:
                data = chart_data["data"]
                
                html_content += f"""
                        </table>
                        
                        <div class="metadata-title" style="margin-top: 20px;">Data</div>
                        <table>
                            <tr>
                """
                
                # Headers
                headers = data[0].keys()
                for header in headers:
                    html_content += f"<th>{header}</th>"
                
                html_content += "</tr>"
                
                # Rows
                for row in data:
                    html_content += "<tr>"
                    for header in headers:
                        cell_value = row.get(header, "")
                        html_content += f"<td>{cell_value}</td>"
                    html_content += "</tr>"
            
            # Close HTML
            html_content += """
                        </table>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error creating HTML: {str(e)}")
            raise
    
    def _export_report_as_pdf(self, report_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Export report as PDF.
        
        Args:
            report_data: Report content and structure
            filename: Base filename without extension
            
        Returns:
            Dictionary with export data
        """
        try:
            # In a real implementation, you would use a PDF library
            # This is a placeholder for the PDF generation logic
            
            buffer = io.BytesIO()
            
            # Placeholder PDF content
            pdf_content = f"""
            %PDF-1.4
            1 0 obj
            << /Type /Catalog
               /Pages 2 0 R
            >>
            endobj
            2 0 obj
            << /Type /Pages
               /Kids [3 0 R]
               /Count 1
            >>
            endobj
            3 0 obj
            << /Type /Page
               /Parent 2 0 R
               /Resources << /Font << /F1 4 0 R >> >>
               /MediaBox [0 0 612 792]
               /Contents 5 0 R
            >>
            endobj
            4 0 obj
            << /Type /Font
               /Subtype /Type1
               /Name /F1
               /BaseFont /Helvetica
            >>
            endobj
            5 0 obj
            << /Length 77 >>
            stream
            BT
            /F1 24 Tf
            72 720 Td
            ({report_data.get("title", "Report")}) Tj
            ET
            endstream
            endobj
            xref
            0 6
            0000000000 65535 f
            0000000009 00000 n
            0000000058 00000 n
            0000000115 00000 n
            0000000234 00000 n
            0000000321 00000 n
            trailer
            << /Size 6
               /Root 1 0 R
            >>
            startxref
            449
            %%EOF
            """
            
            buffer.write(pdf_content.encode("utf-8"))
            buffer.seek(0)
            
            return {
                "format": "pdf",
                "filename": f"{filename}.pdf",
                "mime_type": "application/pdf",
                "data": base64.b64encode(buffer.getvalue()).decode("utf-8")
            }
            
        except Exception as e:
            logger.error(f"Error exporting report as PDF: {str(e)}")
            raise
    
    def _export_report_as_html(self, report_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Export report as HTML.
        
        Args:
            report_data: Report content and structure
            filename: Base filename without extension
            
        Returns:
            Dictionary with export data
        """
        try:
            title = report_data.get("title", "Report")
            description = report_data.get("description", "")
            sections = report_data.get("sections", [])
            
            # Create HTML template
            html_content = f"""<!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .report-title {{ font-size: 28px; font-weight: bold; margin-bottom: 10px; }}
                    .report-description {{ margin-bottom: 30px; color: #555; }}
                    .section {{ margin-bottom: 40px; }}
                    .section-title {{ font-size: 24px; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                    .section-content {{ margin-bottom: 20px; }}
                    .chart-container {{ margin: 20px 0; }}
                    .table-container {{ margin: 20px 0; overflow-x: auto; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="report-title">{title}</h1>
                    <div class="report-description">{description}</div>
            """
            
            # Add sections
            for section in sections:
                section_title = section.get("title", "Section")
                section_content = section.get("content", "")
                section_charts = section.get("charts", [])
                section_tables = section.get("tables", [])
                
                html_content += f"""
                    <div class="section">
                        <h2 class="section-title">{section_title}</h2>
                        <div class="section-content">{section_content}</div>
                """
                
                # Add charts
                for chart in section_charts:
                    chart_title = chart.get("title", "Chart")
                    chart_description = chart.get("description", "")
                    chart_image = chart.get("image", "")
                    
                    # Check image format
                    if chart_image.startswith("<svg"):
                        chart_html = chart_image
                    elif chart_image.startswith("data:image"):
                        chart_html = f'<img src="{chart_image}" alt="{chart_title}" style="max-width:100%;">'
                    else:
                        # Assume base64
                        chart_html = f'<img src="data:image/png;base64,{chart_image}" alt="{chart_title}" style="max-width:100%;">'
                    
                    html_content += f"""
                        <div class="chart-container">
                            <h3>{chart_title}</h3>
                            <p>{chart_description}</p>
                            {chart_html}
                        </div>
                    """
                
                # Add tables
                for table in section_tables:
                    table_title = table.get("title", "Table")
                    table_data = table.get("data", [])
                    
                    html_content += f"""
                        <div class="table-container">
                            <h3>{table_title}</h3>
                            <table>
                    """
                    
                    # Add table headers
                    if table_data and len(table_data) > 0:
                        headers = table_data[0].keys()
                        html_content += "<tr>"
                        for header in headers:
                            html_content += f"<th>{header}</th>"
                        html_content += "</tr>"
                        
                        # Add table rows
                        for row in table_data:
                            html_content += "<tr>"
                            for header in headers:
                                cell_value = row.get(header, "")
                                html_content += f"<td>{cell_value}</td>"
                            html_content += "</tr>"
                    
                    html_content += """
                            </table>
                        </div>
                    """
                
                html_content += """
                    </div>
                """
            
            # Close HTML
            html_content += """
                </div>
            </body>
            </html>
            """
            
            return {
                "format": "html",
                "filename": f"{filename}.html",
                "mime_type": "text/html",
                "data": base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
            }
            
        except Exception as e:
            logger.error(f"Error exporting report as HTML: {str(e)}")
            raise
    
    def _clean_filename(self, filename: str) -> str:
        """
        Clean a filename to make it safe for file systems.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename
        """
        # Remove invalid characters
        clean_name = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Replace spaces with underscores
        clean_name = clean_name.replace(' ', '_')
        
        # Limit length
        if len(clean_name) > 255:
            clean_name = clean_name[:255]
            
        return clean_name
    
    def _json_serializer(self, obj):
        """
        Custom JSON serializer to handle non-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON serializable representation
        """
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)