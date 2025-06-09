// frontend/src/components/forecasting/ForecastDataGrid.jsx
import React, { useState, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-theme-quartz.css';

const ForecastDataGrid = ({ data, selectedCategory }) => {
  const [gridApi, setGridApi] = useState(null);
  
  // Column definitions for the grid
  const columnDefs = useMemo(() => [
    {
      headerName: 'Product Info',
      children: [
        { 
          field: 'product_id', 
          headerName: 'ID', 
          width: 100, 
          pinned: 'left',
          cellClass: 'font-medium'
        },
        { 
          field: 'product_name', 
          headerName: 'Name', 
          width: 200, 
          pinned: 'left' 
        },
        { 
          field: 'category', 
          headerName: 'Category', 
          width: 120 
        },
        { 
          field: 'abc_class', 
          headerName: 'ABC', 
          width: 80,
          cellStyle: (params) => {
            if (params.value === 'A') return { backgroundColor: '#dcfce7', fontWeight: 'bold' };
            if (params.value === 'B') return { backgroundColor: '#fef3c7' };
            return { backgroundColor: '#fee2e2' };
          }
        },
      ],
    },
    {
      headerName: 'Current Stock',
      children: [
        { 
          field: 'current_stock', 
          headerName: 'Quantity', 
          width: 100,
          valueFormatter: (params) => params.value?.toLocaleString() || '0'
        },
        { 
          field: 'stock_value', 
          headerName: 'Value', 
          width: 110,
          valueFormatter: (params) => `$${(params.value || 0).toLocaleString()}`
        },
        { 
          field: 'safety_stock', 
          headerName: 'Safety Stock', 
          width: 110,
          cellStyle: (params) => {
            if (params.data.current_stock < params.value) {
              return { backgroundColor: '#fee2e2', color: '#dc2626' };
            }
            return null;
          }
        },
      ],
    },
    {
      headerName: 'Historical (Last 12 Months)',
      children: Array.from({ length: 12 }, (_, i) => ({
        field: `history_${i}`,
        headerName: `M-${12 - i}`,
        width: 80,
        valueFormatter: (params) => params.value?.toLocaleString() || '0'
      })),
    },
    {
      headerName: 'Forecast (Next 12 Months)',
      children: Array.from({ length: 12 }, (_, i) => ({
        field: `forecast_${i}`,
        headerName: `M+${i + 1}`,
        width: 80,
        editable: true,
        cellStyle: { backgroundColor: '#dbeafe' },
        valueFormatter: (params) => params.value?.toLocaleString() || '0',
        cellEditorParams: {
          min: 0,
          max: 999999,
        },
      })),
    },
    {
      headerName: 'Metrics',
      children: [
        { 
          field: 'mape', 
          headerName: 'MAPE %', 
          width: 90,
          valueFormatter: (params) => `${(params.value || 0).toFixed(2)}%`
        },
        { 
          field: 'trend', 
          headerName: 'Trend', 
          width: 90,
          cellRenderer: (params) => {
            const trend = params.value || 0;
            const icon = trend > 0 ? '↑' : trend < 0 ? '↓' : '→';
            const color = trend > 0 ? 'green' : trend < 0 ? 'red' : 'gray';
            return `<span style="color: ${color}">${icon} ${Math.abs(trend).toFixed(1)}%</span>`;
          }
        },
      ],
    },
  ], []);

  // Generate sample data if not provided
  const rowData = useMemo(() => {
    if (data?.results?.products) {
      return data.results.products;
    }
    
    // Generate sample data
    const categories = ['Electronics', 'Apparel', 'Food & Beverage', 'Home & Garden', 'Sports & Outdoors'];
    const products = [];
    
    for (let i = 0; i < 50; i++) {
      const product = {
        product_id: `SKU${String(i + 1001).padStart(4, '0')}`,
        product_name: `Product ${i + 1}`,
        category: categories[i % categories.length],
        abc_class: i < 10 ? 'A' : i < 30 ? 'B' : 'C',
        current_stock: Math.floor(Math.random() * 5000) + 100,
        stock_value: Math.floor(Math.random() * 50000) + 1000,
        safety_stock: Math.floor(Math.random() * 500) + 50,
        mape: Math.random() * 15 + 2,
        trend: (Math.random() - 0.5) * 20,
      };
      
      // Add historical data
      for (let j = 0; j < 12; j++) {
        product[`history_${j}`] = Math.floor(Math.random() * 1000) + 100;
      }
      
      // Add forecast data
      for (let j = 0; j < 12; j++) {
        product[`forecast_${j}`] = Math.floor(Math.random() * 1200) + 150;
      }
      
      products.push(product);
    }
    
    return products;
  }, [data]);

  // Filter by selected category
  const filteredData = useMemo(() => {
    if (!selectedCategory) return rowData;
    return rowData.filter(row => row.category === selectedCategory);
  }, [rowData, selectedCategory]);

  const onGridReady = (params) => {
    setGridApi(params.api);
  };

  const onCellValueChanged = (params) => {
    console.log('Cell value changed:', params);
    // Here you would typically save the updated value to your backend
  };

  const exportToExcel = () => {
    if (!gridApi) return;
    
    gridApi.exportDataAsCsv({
      fileName: 'forecast_data.csv',
    });
  };

  const defaultColDef = useMemo(() => ({
    sortable: true,
    filter: true,
    resizable: true,
    suppressMenu: true,
  }), []);

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-4 border-b flex justify-between items-center">
        <h3 className="text-lg font-semibold">Forecast Data Grid</h3>
        <button
          onClick={exportToExcel}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
        >
          Export to CSV
        </button>
      </div>
      
      <div className="ag-theme-quartz" style={{ height: '600px', width: '100%' }}>
        <AgGridReact
          rowData={filteredData}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          onGridReady={onGridReady}
          onCellValueChanged={onCellValueChanged}
          animateRows={true}
          rowSelection={{ mode: "multiRow", enableClickSelection: true }}
          suppressCellFocus={true}
          groupDisplayType="groupRows"
        />
      </div>
    </div>
  );
};

export default ForecastDataGrid;
