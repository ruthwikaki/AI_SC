// frontend/src/components/analytics/SupplierDashboard.jsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Loader2, AlertCircle, CheckCircle, XCircle, AlertTriangle, Download } from 'lucide-react';
import api from '../../services/api';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const SupplierDashboard = ({ data }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Analysis states
  const [supplierScorecard, setSupplierScorecard] = useState(null);
  const [riskAnalysis, setRiskAnalysis] = useState(null);
  const [complianceCheck, setComplianceCheck] = useState(null);
  
  // Form states
  const [selectedSupplier, setSelectedSupplier] = useState('');
  const [analysisType, setAnalysisType] = useState('scorecard');
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [suppliers, setSuppliers] = useState([]);

  useEffect(() => {
    fetchSuppliers();
  }, []);

  const fetchSuppliers = async () => {
    try {
      const response = await api.get('/api/analytics/supplier/list');
      // Fix: Extract the suppliers array from the response
      setSuppliers(response.data.suppliers || []);
    } catch (err) {
      console.error('Failed to fetch suppliers:', err);
      setSuppliers([]); // Set empty array on error
    }
  };

  const generateScorecard = async () => {
    if (!selectedSupplier) {
      setError('Please select a supplier');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await api.post('/api/analytics/supplier/scorecard', {
        supplier_id: selectedSupplier,
        start_date: dateRange.start,
        end_date: dateRange.end,
        metrics: ['quality', 'delivery', 'cost', 'responsiveness', 'compliance']
      });
      setSupplierScorecard(response.data);
    } catch (err) {
      setError('Failed to generate scorecard');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const analyzeRisk = async () => {
    if (!selectedSupplier) {
      setError('Please select a supplier');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await api.post('/api/analytics/supplier/risk-analysis', {
        supplier_id: selectedSupplier,
        risk_factors: ['financial', 'operational', 'geopolitical', 'environmental', 'quality']
      });
      setRiskAnalysis(response.data);
    } catch (err) {
      setError('Failed to analyze supplier risk');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const checkCompliance = async () => {
    if (!selectedSupplier) {
      setError('Please select a supplier');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await api.post('/api/analytics/supplier/compliance-check', {
        supplier_id: selectedSupplier,
        compliance_areas: ['certifications', 'quality_standards', 'environmental', 'labor_practices']
      });
      setComplianceCheck(response.data);
    } catch (err) {
      setError('Failed to check compliance');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const exportSupplierReport = async () => {
    if (!selectedSupplier) return;
    
    try {
      const response = await api.get(`/api/analytics/supplier/export/${selectedSupplier}`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `supplier_report_${selectedSupplier}_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      setError('Failed to export report');
    }
  };

  const getRiskBadge = (level) => {
    const variants = {
      low: { color: 'green', icon: CheckCircle },
      medium: { color: 'yellow', icon: AlertTriangle },
      high: { color: 'red', icon: XCircle }
    };
    const variant = variants[level] || variants.medium;
    const Icon = variant.icon;
    
    return (
      <Badge variant={level === 'low' ? 'default' : level === 'medium' ? 'secondary' : 'destructive'}>
        <Icon className="h-3 w-3 mr-1" />
        {level.toUpperCase()}
      </Badge>
    );
  };

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Controls Section */}
      <Card>
        <CardHeader>
          <CardTitle>Supplier Analysis Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Select Supplier</label>
              <Select value={selectedSupplier} onValueChange={setSelectedSupplier}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a supplier" />
                </SelectTrigger>
                <SelectContent>
                  {suppliers && suppliers.length > 0 ? (
                    suppliers.map((supplier) => (
                      <SelectItem key={supplier.id} value={supplier.id.toString()}>
                        {supplier.name} ({supplier.location})
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value="none" disabled>
                      No suppliers available
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-sm font-medium mb-2 block">Start Date</label>
              <Input
                type="date"
                value={dateRange.start}
                onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">End Date</label>
              <Input
                type="date"
                value={dateRange.end}
                onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
              />
            </div>
          </div>

          <div className="mt-4 flex gap-2">
            <Button onClick={generateScorecard} disabled={loading}>
              {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Generate Scorecard
            </Button>
            <Button onClick={analyzeRisk} variant="outline" disabled={loading}>
              Risk Analysis
            </Button>
            <Button onClick={checkCompliance} variant="outline" disabled={loading}>
              Check Compliance
            </Button>
            <Button onClick={exportSupplierReport} variant="outline">
              <Download className="h-4 w-4 mr-2" />
              Export Report
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Supplier List Overview */}
      {suppliers && suppliers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Supplier Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left pb-2">Supplier</th>
                    <th className="text-left pb-2">Category</th>
                    <th className="text-left pb-2">Rating</th>
                    <th className="text-right pb-2">Total Orders</th>
                    <th className="text-right pb-2">Business Value</th>
                  </tr>
                </thead>
                <tbody>
                  {suppliers.slice(0, 5).map((supplier) => (
                    <tr key={supplier.id} className="border-b">
                      <td className="py-2">{supplier.name}</td>
                      <td className="py-2">{supplier.category}</td>
                      <td className="py-2">
                        <span className="font-medium">{supplier.rating}/5</span>
                      </td>
                      <td className="py-2 text-right">{supplier.total_orders}</td>
                      <td className="py-2 text-right">${supplier.total_business.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Supplier Scorecard */}
      {supplierScorecard && (
        <Card>
          <CardHeader>
            <CardTitle>Supplier Performance Scorecard</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-sm font-medium mb-4">Overall Performance</h3>
                <div className="text-center mb-4">
                  <div className="text-5xl font-bold text-blue-600">
                    {supplierScorecard.overall_score?.toFixed(1) || '0'}
                  </div>
                  <p className="text-sm text-gray-600">Out of 100</p>
                  <Badge className="mt-2" variant={
                    supplierScorecard.overall_score >= 80 ? 'default' :
                    supplierScorecard.overall_score >= 60 ? 'secondary' : 'destructive'
                  }>
                    {supplierScorecard.overall_score >= 80 ? 'Excellent' :
                     supplierScorecard.overall_score >= 60 ? 'Good' : 'Needs Improvement'}
                  </Badge>
                </div>

                <div className="space-y-3">
                  {supplierScorecard.metrics && Object.entries(supplierScorecard.metrics).map(([metric, score]) => (
                    <div key={metric} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{metric.replace('_', ' ')}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${score}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium w-12 text-right">{score}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium mb-4">Performance Radar</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={supplierScorecard.metrics ? Object.entries(supplierScorecard.metrics).map(([metric, score]) => ({
                    metric: metric.charAt(0).toUpperCase() + metric.slice(1).replace('_', ' '),
                    score
                  })) : []}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis domain={[0, 100]} />
                    <Radar name="Score" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-medium mb-4">Historical Performance</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={supplierScorecard.historical_data || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="overall" stroke="#3b82f6" name="Overall Score" strokeWidth={2} />
                  <Line type="monotone" dataKey="quality" stroke="#10b981" name="Quality" />
                  <Line type="monotone" dataKey="delivery" stroke="#f59e0b" name="Delivery" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Risk Analysis */}
      {riskAnalysis && (
        <Card>
          <CardHeader>
            <CardTitle>Supplier Risk Assessment</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="text-center p-6 bg-gray-50 rounded">
                <p className="text-sm text-gray-600 mb-2">Overall Risk Level</p>
                {getRiskBadge(riskAnalysis.overall_risk_level || 'medium')}
                <p className="text-3xl font-bold mt-2">{riskAnalysis.risk_score || '0'}/100</p>
              </div>
              <div className="p-4 bg-gray-50 rounded">
                <p className="text-sm font-medium mb-3">Risk Factors</p>
                <div className="space-y-2">
                  {riskAnalysis.risk_factors && Object.entries(riskAnalysis.risk_factors).map(([factor, data]) => (
                    <div key={factor} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{factor.replace('_', ' ')}</span>
                      {getRiskBadge(data.level)}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium mb-4">Risk Factor Analysis</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={riskAnalysis.risk_factors ? Object.entries(riskAnalysis.risk_factors).map(([factor, data]) => ({
                  factor: factor.charAt(0).toUpperCase() + factor.slice(1).replace('_', ' '),
                  score: data.score,
                  threshold: 50
                })) : []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="factor" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="score" fill="#ef4444" />
                  <Bar dataKey="threshold" fill="#fbbf24" opacity={0.3} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {riskAnalysis.mitigation_strategies && riskAnalysis.mitigation_strategies.length > 0 && (
              <div className="mt-6">
                <h3 className="text-sm font-medium mb-3">Recommended Mitigation Strategies</h3>
                <div className="space-y-2">
                  {riskAnalysis.mitigation_strategies.map((strategy, idx) => (
                    <div key={idx} className="flex items-start gap-2 p-3 bg-blue-50 rounded">
                      <CheckCircle className="h-4 w-4 text-blue-600 mt-0.5" />
                      <p className="text-sm">{strategy}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Compliance Check */}
      {complianceCheck && (
        <Card>
          <CardHeader>
            <CardTitle>Compliance Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">Overall Compliance</h3>
                <Badge variant={complianceCheck.is_compliant ? 'default' : 'destructive'}>
                  {complianceCheck.is_compliant ? 'COMPLIANT' : 'NON-COMPLIANT'}
                </Badge>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`h-3 rounded-full ${complianceCheck.compliance_score >= 80 ? 'bg-green-600' : 'bg-yellow-600'}`}
                  style={{ width: `${complianceCheck.compliance_score || 0}%` }}
                />
              </div>
              <p className="text-sm text-gray-600 mt-2">
                Compliance Score: {complianceCheck.compliance_score || 0}%
              </p>
            </div>

            <div className="space-y-4">
              {complianceCheck.compliance_areas && Object.entries(complianceCheck.compliance_areas).map(([area, data]) => (
                <div key={area} className="border rounded p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium capitalize">{area.replace('_', ' ')}</h4>
                    {data.status === 'compliant' ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : data.status === 'partial' ? (
                      <AlertTriangle className="h-5 w-5 text-yellow-600" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-600" />
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{data.description}</p>
                  {data.missing_requirements && data.missing_requirements.length > 0 && (
                    <div className="mt-2">
                      <p className="text-sm font-medium text-red-600">Missing Requirements:</p>
                      <ul className="list-disc list-inside text-sm text-gray-600 mt-1">
                        {data.missing_requirements.map((req, idx) => (
                          <li key={idx}>{req}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {complianceCheck.expiring_certifications && complianceCheck.expiring_certifications.length > 0 && (
              <Alert className="mt-6">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>Expiring Certifications:</strong>
                  <ul className="list-disc list-inside mt-2">
                    {complianceCheck.expiring_certifications.map((cert, idx) => (
                      <li key={idx}>
                        {cert.name} - Expires: {new Date(cert.expiry_date).toLocaleDateString()}
                      </li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default SupplierDashboard;