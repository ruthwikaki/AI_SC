// frontend/src/config/agGridSetup.js
import { ModuleRegistry, AllCommunityModule } from 'ag-grid-community';

// Register only community modules
ModuleRegistry.registerModules([AllCommunityModule]);

// Override any grid options to remove enterprise features
const originalGridOptionsFactory = window.agGridGlobalOptions || {};
window.agGridGlobalOptions = {
  ...originalGridOptionsFactory,
  
  sideBar: undefined,
  enableRangeSelection: undefined,
  masterDetail: undefined
};

// Export safe defaults
export const defaultGridOptions = {
  animateRows: true,
  pagination: true,
  paginationPageSize: 20,
  defaultColDef: {
    sortable: true,
    filter: true,
    resizable: true
  },
  // Explicitly set to undefined to override any defaults
  
  sideBar: undefined
};

export const AG_GRID_THEME = 'ag-theme-quartz';

