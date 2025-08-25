# Diagnóstico de Seguridad del Hogar

Una aplicación web moderna para realizar encuestas sobre seguridad digital y la "Paradoja de la Privacidad" con un diseño elegante en tema oscuro.

## 🎨 Características del Diseño

- **Tema Oscuro Profesional**: Interfaz completamente en modo oscuro con colores modernos y elegantes
- **Gradientes Atractivos**: Fondos con gradientes sutiles que mejoran la experiencia visual
- **Efectos de Hover**: Animaciones suaves en botones y tarjetas
- **Sombras Dinámicas**: Efectos de sombra que dan profundidad a los elementos
- **Tipografía Clara**: Texto optimizado para legibilidad en modo oscuro

## 🚀 Tecnologías Utilizadas

- **Next.js 15**: Framework de React para el frontend
- **TypeScript**: Tipado estático para mayor robustez
- **Tailwind CSS**: Framework de CSS utilitario
- **Radix UI**: Componentes de interfaz accesibles
- **next-themes**: Gestión de temas (modo oscuro)
- **Lucide React**: Iconos modernos y consistentes

## 📋 Funcionalidades

### Encuesta de Seguridad Digital
1. **Información Demográfica**: Edad, educación y experiencia tecnológica
2. **Dispositivos y Conectividad**: Inventario de dispositivos conectados
3. **Comportamientos de Seguridad**: Hábitos de seguridad digital
4. **Paradoja de la Privacidad**: Percepciones sobre privacidad vs conveniencia
5. **Conocimiento de Seguridad**: Nivel de conocimiento sobre amenazas
6. **Comentarios Finales**: Información adicional del participante

### Características Técnicas
- **Navegación por Pasos**: Progreso visual con barra de progreso
- **Validación de Datos**: Verificación de respuestas requeridas
- **Almacenamiento**: API para guardar respuestas de la encuesta
- **Responsive Design**: Optimizado para dispositivos móviles y desktop

## 🎯 Paleta de Colores (Modo Oscuro)

- **Fondo Principal**: `oklch(0.08 0 0)` - Negro muy oscuro
- **Tarjetas**: `oklch(0.12 0.02 240)` - Gris oscuro con tinte azul
- **Primario**: `oklch(0.65 0.25 330)` - Magenta brillante
- **Secundario**: `oklch(0.18 0.05 240)` - Gris oscuro con tinte azul
- **Texto**: `oklch(0.95 0 0)` - Blanco casi puro
- **Bordes**: `oklch(0.22 0.05 240)` - Gris medio con tinte azul

## 🛠️ Instalación y Uso

### Prerrequisitos
- Node.js 18+ 
- npm o pnpm

### Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd security-survey

# Instalar dependencias
npm install
# o
pnpm install

# Ejecutar en modo desarrollo
npm run dev
# o
pnpm dev
```

### Construcción para Producción
```bash
# Construir la aplicación
npm run build

# Iniciar en modo producción
npm start
```

## 📁 Estructura del Proyecto

```
security-survey/
├── app/                    # App Router de Next.js
│   ├── globals.css        # Estilos globales y tema oscuro
│   ├── layout.tsx         # Layout principal con ThemeProvider
│   └── page.tsx           # Página principal de la encuesta
├── components/            # Componentes reutilizables
│   ├── theme-provider.tsx # Proveedor de tema
│   └── ui/               # Componentes de interfaz
├── lib/                  # Utilidades y configuraciones
└── public/               # Archivos estáticos
```

## 🔧 Configuración del Tema Oscuro

El tema oscuro está configurado de forma permanente en:

1. **Layout Principal** (`app/layout.tsx`):
   - Clase `dark` en el elemento `<html>`
   - `ThemeProvider` con `defaultTheme="dark"`

2. **Estilos Globales** (`app/globals.css`):
   - Variables CSS personalizadas para modo oscuro
   - Efectos visuales mejorados
   - Animaciones y transiciones

## 📊 API de Encuesta

La aplicación incluye un endpoint para guardar las respuestas:

- **POST** `/api/survey`
- **Body**: JSON con los datos de la encuesta
- **Respuesta**: Confirmación de guardado exitoso

## 🎨 Personalización

### Modificar Colores
Los colores se pueden personalizar editando las variables CSS en `app/globals.css`:

```css
.dark {
  --background: oklch(0.08 0 0); /* Fondo principal */
  --primary: oklch(0.65 0.25 330); /* Color primario */
  /* ... más variables */
}
```

### Agregar Nuevos Pasos
Para agregar nuevos pasos a la encuesta, modificar el array `totalSteps` y agregar el caso correspondiente en `renderStep()`.

## 📱 Responsive Design

La aplicación está optimizada para:
- **Desktop**: 1024px+
- **Tablet**: 768px - 1023px
- **Mobile**: < 768px

## 🔒 Seguridad

- Validación de datos en el frontend
- Sanitización de inputs
- Respuestas anónimas
- Uso seguro de APIs

## 📄 Licencia

Este proyecto es parte de una investigación académica sobre la "Paradoja de la Privacidad".

## 🤝 Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

---

**Desarrollado por Huaritex para la investigación sobre seguridad digital**
