"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { Shield, Home, Lock, Wifi, Eye, ChevronRight, ChevronLeft } from "lucide-react"

interface SurveyData {
  // Datos demográficos
  age: string
  gender: string
  education: string
  techExperience: string

  // Dispositivos y conectividad
  devices: string[]
  internetProvider: string
  routerAge: string
  wifiPassword: string

  // Comportamientos de seguridad
  passwordManager: string
  twoFactor: string
  softwareUpdates: string
  publicWifi: string

  // Paradoja de la privacidad
  privacyConcern: string
  dataSharing: string
  convenienceVsPrivacy: string
  trustInTech: string

  // Conocimiento de seguridad
  securityKnowledge: string
  threatAwareness: string[]
  securityMeasures: string[]

  // Comentarios adicionales
  additionalComments: string
}

const initialData: SurveyData = {
  age: "",
  gender: "",
  education: "",
  techExperience: "",
  devices: [],
  internetProvider: "",
  routerAge: "",
  wifiPassword: "",
  passwordManager: "",
  twoFactor: "",
  softwareUpdates: "",
  publicWifi: "",
  privacyConcern: "",
  dataSharing: "",
  convenienceVsPrivacy: "",
  trustInTech: "",
  securityKnowledge: "",
  threatAwareness: [],
  securityMeasures: [],
  additionalComments: "",
}

export default function HomeSurveyPage() {
  const [currentStep, setCurrentStep] = useState(0)
  const [surveyData, setSurveyData] = useState<SurveyData>(initialData)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const totalSteps = 6
  const progress = ((currentStep + 1) / totalSteps) * 100

  const handleNext = () => {
    if (currentStep < totalSteps - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleSubmit = async () => {
    setIsSubmitting(true)

    try {
      const response = await fetch("/api/survey", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(surveyData),
      })

      if (response.ok) {
        alert("¡Encuesta enviada exitosamente! Gracias por tu participación.")
        setSurveyData(initialData)
        setCurrentStep(0)
      } else {
        alert("Error al enviar la encuesta. Por favor, inténtalo de nuevo.")
      }
    } catch (error) {
      console.error("Error:", error)
      alert("Error al enviar la encuesta. Por favor, inténtalo de nuevo.")
    } finally {
      setIsSubmitting(false)
    }
  }

  const updateData = (field: keyof SurveyData, value: any) => {
    setSurveyData((prev) => ({ ...prev, [field]: value }))
  }

  const updateArrayData = (field: keyof SurveyData, value: string, checked: boolean) => {
    setSurveyData((prev) => {
      const currentArray = prev[field] as string[]
      if (checked) {
        return { ...prev, [field]: [...currentArray, value] }
      } else {
        return { ...prev, [field]: currentArray.filter((item) => item !== value) }
      }
    })
  }

  const renderStep = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="space-y-6 animate-fade-in">
            <div className="text-center space-y-4">
              <div className="flex justify-center">
                <div className="p-4 bg-primary/10 rounded-full">
                  <Shield className="h-12 w-12 text-primary" />
                </div>
              </div>
              <h2 className="text-2xl font-bold text-foreground">Diagnóstico de Seguridad del Hogar</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Esta herramienta es parte de una investigación académica sobre la "Paradoja de la Privacidad".
                Realizaremos una <strong>simulación</strong> para evaluar la seguridad de una red doméstica.
              </p>
            </div>

            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="h-5 w-5 text-primary" />
                  Información Demográfica
                </CardTitle>
                <CardDescription>Ayúdanos a entender mejor el perfil de los participantes</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="age">Rango de edad</Label>
                    <RadioGroup value={surveyData.age} onValueChange={(value) => updateData("age", value)}>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="18-25" id="age1" />
                        <Label htmlFor="age1">18-25 años</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="26-35" id="age2" />
                        <Label htmlFor="age2">26-35 años</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="36-45" id="age3" />
                        <Label htmlFor="age3">36-45 años</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="46+" id="age4" />
                        <Label htmlFor="age4">46+ años</Label>
                      </div>
                    </RadioGroup>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="education">Nivel educativo</Label>
                    <RadioGroup value={surveyData.education} onValueChange={(value) => updateData("education", value)}>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="secundaria" id="edu1" />
                        <Label htmlFor="edu1">Secundaria</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="tecnico" id="edu2" />
                        <Label htmlFor="edu2">Técnico</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="universitario" id="edu3" />
                        <Label htmlFor="edu3">Universitario</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="posgrado" id="edu4" />
                        <Label htmlFor="edu4">Posgrado</Label>
                      </div>
                    </RadioGroup>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="techExperience">¿Cómo calificarías tu experiencia con tecnología?</Label>
                  <RadioGroup
                    value={surveyData.techExperience}
                    onValueChange={(value) => updateData("techExperience", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="principiante" id="tech1" />
                      <Label htmlFor="tech1">Principiante</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="intermedio" id="tech2" />
                      <Label htmlFor="tech2">Intermedio</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="avanzado" id="tech3" />
                      <Label htmlFor="tech3">Avanzado</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="experto" id="tech4" />
                      <Label htmlFor="tech4">Experto</Label>
                    </div>
                  </RadioGroup>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 1:
        return (
          <div className="space-y-6 animate-fade-in">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Home className="h-5 w-5 text-primary" />
                  Dispositivos y Conectividad
                </CardTitle>
                <CardDescription>Información sobre tus dispositivos conectados y red doméstica</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>
                    ¿Qué dispositivos tienes conectados a internet en casa? (Selecciona todos los que apliquen)
                  </Label>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      "Smartphone",
                      "Laptop/PC",
                      "Smart TV",
                      "Tablet",
                      "Consola de videojuegos",
                      "Dispositivos IoT (Alexa, Google Home)",
                      "Cámaras de seguridad",
                      "Otros",
                    ].map((device) => (
                      <div key={device} className="flex items-center space-x-2">
                        <Checkbox
                          id={device}
                          checked={surveyData.devices.includes(device)}
                          onCheckedChange={(checked) => updateArrayData("devices", device, checked as boolean)}
                        />
                        <Label htmlFor={device} className="text-sm">
                          {device}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="routerAge">¿Qué edad tiene tu router/módem?</Label>
                  <RadioGroup value={surveyData.routerAge} onValueChange={(value) => updateData("routerAge", value)}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="menos-1" id="router1" />
                      <Label htmlFor="router1">Menos de 1 año</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="1-3" id="router2" />
                      <Label htmlFor="router2">1-3 años</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="3-5" id="router3" />
                      <Label htmlFor="router3">3-5 años</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="mas-5" id="router4" />
                      <Label htmlFor="router4">Más de 5 años</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="wifiPassword">¿Qué tipo de contraseña usas para tu WiFi?</Label>
                  <RadioGroup
                    value={surveyData.wifiPassword}
                    onValueChange={(value) => updateData("wifiPassword", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="predeterminada" id="wifi1" />
                      <Label htmlFor="wifi1">La que venía por defecto</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="simple" id="wifi2" />
                      <Label htmlFor="wifi2">Una contraseña simple (nombre, fecha, etc.)</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="compleja" id="wifi3" />
                      <Label htmlFor="wifi3">Una contraseña compleja y única</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no-se" id="wifi4" />
                      <Label htmlFor="wifi4">No lo sé</Label>
                    </div>
                  </RadioGroup>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 2:
        return (
          <div className="space-y-6 animate-fade-in">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lock className="h-5 w-5 text-primary" />
                  Comportamientos de Seguridad
                </CardTitle>
                <CardDescription>Tus hábitos actuales de seguridad digital</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="passwordManager">¿Usas un gestor de contraseñas?</Label>
                  <RadioGroup
                    value={surveyData.passwordManager}
                    onValueChange={(value) => updateData("passwordManager", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="si-siempre" id="pm1" />
                      <Label htmlFor="pm1">Sí, siempre</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="a-veces" id="pm2" />
                      <Label htmlFor="pm2">A veces</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="pm3" />
                      <Label htmlFor="pm3">No</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no-se-que-es" id="pm4" />
                      <Label htmlFor="pm4">No sé qué es eso</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="twoFactor">¿Usas autenticación de dos factores (2FA)?</Label>
                  <RadioGroup value={surveyData.twoFactor} onValueChange={(value) => updateData("twoFactor", value)}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="si-siempre" id="2fa1" />
                      <Label htmlFor="2fa1">Sí, en todas las cuentas importantes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="algunas" id="2fa2" />
                      <Label htmlFor="2fa2">Solo en algunas cuentas</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="2fa3" />
                      <Label htmlFor="2fa3">No</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no-se-que-es" id="2fa4" />
                      <Label htmlFor="2fa4">No sé qué es eso</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="softwareUpdates">
                    ¿Con qué frecuencia actualizas el software de tus dispositivos?
                  </Label>
                  <RadioGroup
                    value={surveyData.softwareUpdates}
                    onValueChange={(value) => updateData("softwareUpdates", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="inmediatamente" id="update1" />
                      <Label htmlFor="update1">Inmediatamente cuando están disponibles</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="semanalmente" id="update2" />
                      <Label htmlFor="update2">Semanalmente</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="mensualmente" id="update3" />
                      <Label htmlFor="update3">Mensualmente</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="raramente" id="update4" />
                      <Label htmlFor="update4">Raramente o nunca</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="publicWifi">¿Cómo usas las redes WiFi públicas?</Label>
                  <RadioGroup value={surveyData.publicWifi} onValueChange={(value) => updateData("publicWifi", value)}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="nunca" id="wifi-pub1" />
                      <Label htmlFor="wifi-pub1">Nunca las uso</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="con-vpn" id="wifi-pub2" />
                      <Label htmlFor="wifi-pub2">Las uso con VPN</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="navegacion-basica" id="wifi-pub3" />
                      <Label htmlFor="wifi-pub3">Solo para navegación básica</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="todo" id="wifi-pub4" />
                      <Label htmlFor="wifi-pub4">Para todo tipo de actividades</Label>
                    </div>
                  </RadioGroup>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 3:
        return (
          <div className="space-y-6 animate-fade-in">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="h-5 w-5 text-primary" />
                  Paradoja de la Privacidad
                </CardTitle>
                <CardDescription>Tus percepciones sobre privacidad y conveniencia</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="privacyConcern">¿Qué tan preocupado estás por tu privacidad en línea?</Label>
                  <RadioGroup
                    value={surveyData.privacyConcern}
                    onValueChange={(value) => updateData("privacyConcern", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="muy-preocupado" id="privacy1" />
                      <Label htmlFor="privacy1">Muy preocupado</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="algo-preocupado" id="privacy2" />
                      <Label htmlFor="privacy2">Algo preocupado</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="poco-preocupado" id="privacy3" />
                      <Label htmlFor="privacy3">Poco preocupado</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="nada-preocupado" id="privacy4" />
                      <Label htmlFor="privacy4">Nada preocupado</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="dataSharing">¿Compartes datos personales a cambio de servicios gratuitos?</Label>
                  <RadioGroup
                    value={surveyData.dataSharing}
                    onValueChange={(value) => updateData("dataSharing", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="nunca" id="data1" />
                      <Label htmlFor="data1">Nunca</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="raramente" id="data2" />
                      <Label htmlFor="data2">Raramente</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="a-veces" id="data3" />
                      <Label htmlFor="data3">A veces</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="frecuentemente" id="data4" />
                      <Label htmlFor="data4">Frecuentemente</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="convenienceVsPrivacy">¿Qué es más importante para ti?</Label>
                  <RadioGroup
                    value={surveyData.convenienceVsPrivacy}
                    onValueChange={(value) => updateData("convenienceVsPrivacy", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="privacidad" id="conv1" />
                      <Label htmlFor="conv1">Privacidad absoluta</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="balance" id="conv2" />
                      <Label htmlFor="conv2">Balance entre privacidad y conveniencia</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="conveniencia" id="conv3" />
                      <Label htmlFor="conv3">Conveniencia sobre privacidad</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="trustInTech">¿Qué tanto confías en las empresas tecnológicas con tus datos?</Label>
                  <RadioGroup
                    value={surveyData.trustInTech}
                    onValueChange={(value) => updateData("trustInTech", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="mucho" id="trust1" />
                      <Label htmlFor="trust1">Confío mucho</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="algo" id="trust2" />
                      <Label htmlFor="trust2">Confío algo</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="poco" id="trust3" />
                      <Label htmlFor="trust3">Confío poco</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="nada" id="trust4" />
                      <Label htmlFor="trust4">No confío nada</Label>
                    </div>
                  </RadioGroup>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 4:
        return (
          <div className="space-y-6 animate-fade-in">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-primary" />
                  Conocimiento de Seguridad
                </CardTitle>
                <CardDescription>Tu nivel de conocimiento sobre amenazas y medidas de seguridad</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="securityKnowledge">
                    ¿Cómo calificarías tu conocimiento sobre seguridad informática?
                  </Label>
                  <RadioGroup
                    value={surveyData.securityKnowledge}
                    onValueChange={(value) => updateData("securityKnowledge", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="experto" id="know1" />
                      <Label htmlFor="know1">Experto</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="avanzado" id="know2" />
                      <Label htmlFor="know2">Avanzado</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="intermedio" id="know3" />
                      <Label htmlFor="know3">Intermedio</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="principiante" id="know4" />
                      <Label htmlFor="know4">Principiante</Label>
                    </div>
                  </RadioGroup>
                </div>

                <div className="space-y-2">
                  <Label>¿Cuáles de estas amenazas conoces? (Selecciona todas las que apliquen)</Label>
                  <div className="grid grid-cols-1 gap-2">
                    {[
                      "Phishing",
                      "Malware",
                      "Ransomware",
                      "Ataques de fuerza bruta",
                      "Man-in-the-middle",
                      "Ingeniería social",
                      "Ataques DDoS",
                      "Ninguna",
                    ].map((threat) => (
                      <div key={threat} className="flex items-center space-x-2">
                        <Checkbox
                          id={threat}
                          checked={surveyData.threatAwareness.includes(threat)}
                          onCheckedChange={(checked) => updateArrayData("threatAwareness", threat, checked as boolean)}
                        />
                        <Label htmlFor={threat} className="text-sm">
                          {threat}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>¿Qué medidas de seguridad implementas en casa? (Selecciona todas las que apliquen)</Label>
                  <div className="grid grid-cols-1 gap-2">
                    {[
                      "Antivirus actualizado",
                      "Firewall activado",
                      "Red de invitados separada",
                      "Cambio regular de contraseñas",
                      "Monitoreo de red",
                      "Copias de seguridad regulares",
                      "VPN doméstica",
                      "Ninguna",
                    ].map((measure) => (
                      <div key={measure} className="flex items-center space-x-2">
                        <Checkbox
                          id={measure}
                          checked={surveyData.securityMeasures.includes(measure)}
                          onCheckedChange={(checked) =>
                            updateArrayData("securityMeasures", measure, checked as boolean)
                          }
                        />
                        <Label htmlFor={measure} className="text-sm">
                          {measure}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 5:
        return (
          <div className="space-y-6 animate-fade-in">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wifi className="h-5 w-5 text-primary" />
                  Comentarios Finales
                </CardTitle>
                <CardDescription>Comparte cualquier información adicional que consideres relevante</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="additionalComments">
                    ¿Hay algo más que te gustaría compartir sobre tus hábitos de seguridad digital o preocupaciones
                    sobre privacidad?
                  </Label>
                  <Textarea
                    id="additionalComments"
                    placeholder="Escribe tus comentarios aquí..."
                    value={surveyData.additionalComments}
                    onChange={(e) => updateData("additionalComments", e.target.value)}
                    className="min-h-[120px]"
                  />
                </div>

                <div className="bg-muted/50 p-4 rounded-lg">
                  <h3 className="font-semibold text-foreground mb-2">Información sobre el estudio</h3>
                  <p className="text-sm text-muted-foreground">
                    Esta encuesta es parte de una investigación académica sobre la "Paradoja de la Privacidad" - el
                    fenómeno donde las personas expresan preocupación por su privacidad pero continúan compartiendo
                    información personal. Tus respuestas son anónimas y serán utilizadas únicamente para fines
                    académicos.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted/20 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Diagnóstico de Seguridad del Hogar</h1>
          <p className="text-muted-foreground">Investigación sobre la Paradoja de la Privacidad</p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between text-sm text-muted-foreground mb-2">
            <span>
              Paso {currentStep + 1} de {totalSteps}
            </span>
            <span>{Math.round(progress)}% completado</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Survey Content */}
        <div className="mb-8">{renderStep()}</div>

        {/* Navigation */}
        <div className="flex justify-between">
          <Button
            variant="outline"
            onClick={handlePrevious}
            disabled={currentStep === 0}
            className="flex items-center gap-2 bg-transparent"
          >
            <ChevronLeft className="h-4 w-4" />
            Anterior
          </Button>

          {currentStep === totalSteps - 1 ? (
            <Button onClick={handleSubmit} disabled={isSubmitting} className="flex items-center gap-2">
              {isSubmitting ? "Enviando..." : "Enviar Encuesta"}
              <Shield className="h-4 w-4" />
            </Button>
          ) : (
            <Button onClick={handleNext} className="flex items-center gap-2">
              Siguiente
              <ChevronRight className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
