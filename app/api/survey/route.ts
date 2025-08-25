import { type NextRequest, NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function POST(request: NextRequest) {
  try {
    const surveyData = await request.json()

    // Agregar timestamp y ID único
    const enrichedData = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      ...surveyData,
    }

    // Ruta del archivo de datos
    const dataDir = path.join(process.cwd(), "data")
    const dataFile = path.join(dataDir, "survey_responses.json")

    // Crear directorio si no existe
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true })
    }

    // Leer datos existentes o crear array vacío
    let existingData = []
    if (fs.existsSync(dataFile)) {
      const fileContent = fs.readFileSync(dataFile, "utf8")
      existingData = JSON.parse(fileContent)
    }

    // Agregar nueva respuesta
    existingData.push(enrichedData)

    // Guardar datos actualizados
    fs.writeFileSync(dataFile, JSON.stringify(existingData, null, 2))

    // También guardar en formato CSV para análisis
    const csvFile = path.join(dataDir, "survey_responses.csv")
    const csvHeaders = Object.keys(enrichedData).join(",")
    const csvRow = Object.values(enrichedData)
      .map((value) => (Array.isArray(value) ? `"${value.join(";")}"` : `"${value}"`))
      .join(",")

    if (!fs.existsSync(csvFile)) {
      fs.writeFileSync(csvFile, csvHeaders + "\n")
    }
    fs.appendFileSync(csvFile, csvRow + "\n")

    return NextResponse.json({
      success: true,
      message: "Encuesta guardada exitosamente",
      id: enrichedData.id,
    })
  } catch (error) {
    console.error("Error saving survey:", error)
    return NextResponse.json({ success: false, message: "Error al guardar la encuesta" }, { status: 500 })
  }
}
