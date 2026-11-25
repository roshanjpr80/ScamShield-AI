'use client'

import { cn } from '@/lib/utils'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts'
import { formatNumber, formatLatency } from '@/lib/utils'

const COLORS = ['#0ea5e9', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

interface TrendChartProps {
  data: Array<{ timestamp: string; transactions: number; fraud_detected: number; avg_latency_ms: number }>
  className?: string
}

export function TrendChart({ data, className }: TrendChartProps) {
  if (!data?.length) return <div className={cn('h-64', className)}>No data available</div>

  const formattedData = data.map((d) => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
  }))

  return (
    <div className={cn('h-64', className)}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={formattedData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} tickCount={8} />
          <YAxis yAxisId="left" stroke="#9ca3af" fontSize={11} tickCount={5} />
          <YAxis yAxisId="right" orientation="right" stroke="#9ca3af" fontSize={11} tickCount={5} />
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
            formatter={(value: number, name: string) => {
              if (name === 'avg_latency_ms') return [formatLatency(value), name]
              if (name === 'transactions') return [formatNumber(value), name]
              return [value, name]
            }}
          />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="transactions"
            stroke={COLORS[0]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
            name="Transactions"
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="fraud_detected"
            stroke={COLORS[3]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
            name="Fraud Detected"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="avg_latency_ms"
            stroke={COLORS[2]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
            name="Avg Latency (ms)"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

interface DecisionPieProps {
  data: Array<{ name: string; value: number }>
  className?: string
}

export function DecisionPie({ data, className }: DecisionPieProps) {
  if (!data?.length) return <div className={cn('h-64', className)}>No data available</div>

  const total = data.reduce((sum, d) => sum + d.value, 0)

  return (
    <div className={cn('h-64', className)}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            fill="#8884d8"
            paddingAngle={2}
            dataKey="value"
            nameKey="name"
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
            labelLine={false}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            formatter={(value: number) => [formatNumber(value), 'Transactions']}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap justify-center gap-4 mt-4 text-sm">
        {data.map((entry, index) => (
          <div key={entry.name} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: COLORS[index % COLORS.length] }}
            />
            <span className="text-gray-700">
              {entry.name}: {formatNumber(entry.value)} ({(entry.value / total * 100).toFixed(1)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

interface LatencyChartProps {
  data: Array<{ time: string; p50: number; p95: number; p99: number }>
  className?: string
}

export function LatencyChart({ data, className }: LatencyChartProps) {
  if (!data?.length) return <div className={cn('h-64', className)}>No data available</div>

  return (
    <div className={cn('h-64', className)}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} tickCount={8} />
          <YAxis stroke="#9ca3af" fontSize={11} tickCount={5} tickFormatter={(v) => formatLatency(v)} />
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            formatter={(value: number) => [formatLatency(value), 'Latency']}
          />
          <Legend />
          <Area type="monotone" dataKey="p50" stroke={COLORS[1]} fill={COLORS[1]} fillOpacity={0.1} name="P50" />
          <Area type="monotone" dataKey="p95" stroke={COLORS[2]} fill={COLORS[2]} fillOpacity={0.1} name="P95" />
          <Area type="monotone" dataKey="p99" stroke={COLORS[3]} fill={COLORS[3]} fillOpacity={0.1} name="P99" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

interface FraudTypeBarProps {
  data: Array<{ type: string; count: number }>
  className?: string
}

export function FraudTypeBar({ data, className }: FraudTypeBarProps) {
  if (!data?.length) return <div className={cn('h-64', className)}>No data available</div>

  return (
    <div className={cn('h-64', className)}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis type="number" stroke="#9ca3af" fontSize={11} tickFormatter={(v) => formatNumber(v)} />
          <YAxis dataKey="type" type="category" stroke="#9ca3af" fontSize={11} width={120} />
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            formatter={(value: number) => [formatNumber(value), 'Count']}
          />
          <Legend />
          <Bar dataKey="count" fill={COLORS[0]} radius={[0, 4, 4, 0]} name="Count" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface GeographyChartProps {
  data: Array<{ state: string; transactions: number; fraud_rate: number }>
  className?: string
}

export function GeographyChart({ data, className }: GeographyChartProps) {
  if (!data?.length) return <div className={cn('h-64', className)}>No data available</div>

  return (
    <div className={cn('h-64', className)}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data.sort((a, b) => b.transactions - a.transactions)}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
          <XAxis type="number" stroke="#9ca3af" fontSize={11} tickFormatter={(v) => formatNumber(v)} />
          <YAxis dataKey="state" type="category" stroke="#9ca3af" fontSize={11} width={100} />
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            formatter={(value: number, name: string) => {
              if (name === 'fraud_rate') return [`${(value * 100).toFixed(2)}%`, name]
              return [formatNumber(value), name]
            }}
          />
          <Legend />
          <Bar dataKey="transactions" fill={COLORS[0]} radius={[4, 0, 0, 4]} name="Transactions" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}