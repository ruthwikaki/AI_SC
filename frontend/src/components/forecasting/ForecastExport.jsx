import React from 'react';

const ForecastExport = ({ data }) => {
  const exportToCSV = () => {
    // Implementation for CSV export
    console.log('Exporting to CSV:', data);
  };

  const exportToExcel = () => {
    // Implementation for Excel export
    console.log('Exporting to Excel:', data);
  };

  const exportToPDF = () => {
    // Implementation for PDF export
    console.log('Exporting to PDF:', data);
  };

  return (
    <div className="flex space-x-2">
      <button
        onClick={exportToCSV}
        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
      >
        Export CSV
      </button>
      <button
        onClick={exportToExcel}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Export Excel
      </button>
      <button
        onClick={exportToPDF}
        className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
      >
        Export PDF
      </button>
    </div>
  );
};

export default ForecastExport;
