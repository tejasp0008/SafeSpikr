import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, CheckCircle, Clock, Brain, Shield, Zap } from "lucide-react";

export function PredictionPanel() {
  const rawPrediction = "Safe";
  const confidenceScore = 92;
  const riskLevel = "Low";

  return (
    <div className="space-y-6">
      {/* Neural Prediction */}
      <Card className="futuristic-card neural-glow hover-float animate-fade-in">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center space-x-3 font-display">
            <Brain className="h-6 w-6 text-accent" />
            <span className="text-neural">Neural Prediction</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-6">
            <div className="space-y-2">
              <div className="text-4xl font-bold font-display text-gradient">{rawPrediction}</div>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-5 w-5 text-green-500 animate-glow" />
                <span className="text-sm text-muted-foreground font-medium">Neural Confidence: {confidenceScore}%</span>
              </div>
            </div>
            <div className="relative">
              <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center shadow-neural">
                <Shield className="h-8 w-8 text-primary-foreground" />
              </div>
              <div className="absolute inset-0 bg-gradient-primary rounded-full animate-pulse-ring opacity-20" />
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">AI Processing</span>
              <span className="text-muted-foreground">{confidenceScore}%</span>
            </div>
            <Progress value={confidenceScore} className="h-3 bg-muted/50" />
          </div>
        </CardContent>
      </Card>

      {/* Personalized Analysis */}
      <Card className="futuristic-card hover-float animate-fade-in" style={{ animationDelay: "0.1s" }}>
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center space-x-3 font-display">
            <Zap className="h-6 w-6 text-accent" />
            <span>Personalized Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Risk Assessment</span>
            <Badge 
              variant={riskLevel === "Low" ? "secondary" : "destructive"} 
              className={riskLevel === "Low" ? "animate-glow" : ""}
            >
              {riskLevel} Risk
            </Badge>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">Safety Score</span>
              <span className="text-muted-foreground font-bold">{confidenceScore}/100</span>
            </div>
            <Progress value={confidenceScore} className="h-3 bg-muted/50" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 rounded-xl bg-gradient-subtle border border-white/10 hover-float">
              <div className="text-2xl font-bold text-green-600 font-display">85%</div>
              <div className="text-xs text-muted-foreground font-medium mt-1">Attention</div>
            </div>
            <div className="text-center p-4 rounded-xl bg-gradient-subtle border border-white/10 hover-float">
              <div className="text-2xl font-bold text-accent font-display">15%</div>
              <div className="text-xs text-muted-foreground font-medium mt-1">Fatigue</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Alert System */}
      <Card className="futuristic-card hover-float animate-fade-in" style={{ animationDelay: "0.2s" }}>
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center space-x-3 font-display">
            <AlertTriangle className="h-6 w-6 text-accent" />
            <span>Alert System</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center space-x-4 p-4 rounded-xl bg-gradient-subtle border border-white/10">
              <div className="p-2 bg-green-100 rounded-full">
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
              <div className="flex-1">
                <div className="text-sm font-semibold">All Systems Nominal</div>
                <div className="text-xs text-muted-foreground">Neural monitoring active</div>
              </div>
              <Badge variant="secondary" className="animate-glow">
                ACTIVE
              </Badge>
            </div>
            
            <div className="text-center py-6">
              <Clock className="h-8 w-8 text-muted-foreground mx-auto mb-2 opacity-50" />
              <p className="text-sm text-muted-foreground">No recent incidents</p>
              <p className="text-xs text-muted-foreground/70 mt-1">Keep up the excellent driving!</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}