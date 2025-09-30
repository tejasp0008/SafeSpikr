import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CameraFeed } from "./CameraFeed";
import { PredictionPanel } from "./PredictionPanel";
import { 
  Car, 
  Shield, 
  TrendingUp, 
  Clock, 
  AlertTriangle,
  CheckCircle,
  Eye,
  Activity,
  Brain,
  Zap,
  Wifi,
  Cpu
} from "lucide-react";

export function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="p-8 space-y-8">
        {/* Header */}
        <div className="mb-10 animate-fade-in">
          <div className="flex items-center space-x-4 mb-4">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-primary rounded-2xl flex items-center justify-center shadow-neural">
                <Brain className="h-6 w-6 text-primary-foreground" />
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full animate-glow" />
            </div>
            <div>
              <h1 className="text-4xl font-bold font-display text-gradient">
                SafeSpikr Dashboard
              </h1>
              <p className="text-muted-foreground text-lg font-medium mt-1">
                Advanced AI-powered driver behavior monitoring 
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <Badge variant="secondary" className="animate-glow">
              <Wifi className="h-3 w-3 mr-1" />
              NEURAL LINK ACTIVE
            </Badge>
            <Badge variant="outline">
              <Cpu className="h-3 w-3 mr-1" />
              AI Processing: 99.7%
            </Badge>
          </div>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
          <Card className="futuristic-card neural-glow hover-float animate-slide-up">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">System Status</p>
                  <p className="text-3xl font-bold font-display text-green-600">Safe</p>
                </div>
                <div className="relative">
                  <div className="p-4 bg-green-100 rounded-2xl">
                    <Shield className="h-8 w-8 text-green-600" />
                  </div>
                  <div className="absolute inset-0 bg-green-500/20 rounded-2xl animate-pulse-ring" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="futuristic-card hover-float animate-slide-up" style={{ animationDelay: "0.1s" }}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Neural State</p>
                  <p className="text-3xl font-bold font-display text-accent">Active</p>
                </div>
                <div className="p-4 bg-blue-100 rounded-2xl">
                  <Activity className="h-8 w-8 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="futuristic-card hover-float animate-slide-up" style={{ animationDelay: "0.2s" }}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Safety Score</p>
                  <p className="text-3xl font-bold font-display text-gradient">92%</p>
                </div>
                <div className="p-4 bg-orange-100 rounded-2xl">
                  <TrendingUp className="h-8 w-8 text-orange-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="futuristic-card hover-float animate-slide-up" style={{ animationDelay: "0.3s" }}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Session Time</p>
                  <p className="text-3xl font-bold font-display text-purple-600">2h 15m</p>
                </div>
                <div className="p-4 bg-purple-100 rounded-2xl">
                  <Clock className="h-8 w-8 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-10">
          {/* Camera Feed - Takes 2/3 of the space */}
          <div className="lg:col-span-2">
            <CameraFeed />
          </div>

          {/* Prediction Panel - Takes 1/3 of the space */}
          <div className="lg:col-span-1">
            <PredictionPanel />
          </div>
        </div>

        {/* Enhanced Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Card className="futuristic-card neural-glow hover-float animate-fade-in">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center space-x-3 font-display">
                <Eye className="h-6 w-6 text-accent" />
                <span className="text-gradient">Neural Detection Matrix</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-subtle border border-white/10">
                  <span className="text-sm font-medium">Eye Tracking</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-glow" />
                    <Badge variant="secondary">98.5%</Badge>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-subtle border border-white/10">
                  <span className="text-sm font-medium">Attention Level</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-glow" />
                    <Badge variant="secondary">High</Badge>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-subtle border border-white/10">
                  <span className="text-sm font-medium">Fatigue Detection</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-glow" />
                    <Badge variant="secondary">Optimal</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="futuristic-card hover-float animate-fade-in" style={{ animationDelay: "0.1s" }}>
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center space-x-3 font-display">
                <Car className="h-6 w-6 text-accent" />
                <span className="text-gradient">Vehicle Integration</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-subtle border border-white/10">
                  <span className="text-sm font-medium">Speed Monitoring</span>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500 animate-glow" />
                    <span className="text-sm text-muted-foreground">Active</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-subtle border border-white/10">
                  <span className="text-sm font-medium">Lane Detection</span>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500 animate-glow" />
                    <span className="text-sm text-muted-foreground">Active</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-subtle border border-white/10">
                  <span className="text-sm font-medium">Emergency Response</span>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500 animate-glow" />
                    <span className="text-sm text-muted-foreground">Standby</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}