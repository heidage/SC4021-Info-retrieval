import React from 'react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { StockData } from '../types';

interface DataTableProps {
  data: StockData[];
}

const columnHelper = createColumnHelper<StockData>();

const columns = [
  columnHelper.accessor('symbol', {
    header: 'Symbol',
    cell: info => info.getValue(),
  }),
  columnHelper.accessor('timestamp', {
    header: 'Timestamp',
    cell: info => new Date(info.getValue()).toLocaleString(),
  }),
  columnHelper.accessor('subreddits', {
    header: 'Subreddits',
    cell: info => info.getValue().join(', '),
  }),
  columnHelper.accessor('comments', {
    header: 'Posts Found',
    cell: info => info.getValue().length,
  }),
];

export const DataTable: React.FC<DataTableProps> = ({ data }) => {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div className="overflow-x-auto">
      <div className="text-sm mb-2">Total Searches: {data.length}</div>
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          {table.getHeaderGroups().map(headerGroup => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map(header => (
                <th
                  key={header.id}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {flexRender(
                    header.column.columnDef.header,
                    header.getContext()
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {table.getRowModel().rows.map(row => (
            <tr key={row.id}>
              {row.getVisibleCells().map(cell => (
                <td
                  key={cell.id}
                  className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};