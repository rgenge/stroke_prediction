# Static Files Structure

This folder contains all static assets for the Stroke Prediction web application.

## Folder Structure

```
static/
├── css/
│   └── style.css          # Main application styles
├── js/
│   └── app.js            # Main application JavaScript
├── images/               # Image assets (logos, icons, etc.)
└── README.md            # This file
```

## File Descriptions

### CSS Files
- **style.css**: Contains all the styling for the web application including:
  - Layout and typography
  - Upload area styling
  - Button and form styles
  - Loading animations
  - Responsive design rules

### JavaScript Files
- **app.js**: Contains all client-side functionality including:
  - File upload handling
  - Drag and drop functionality
  - API communication for predictions
  - UI state management
  - Error handling

### Images Folder
- Currently empty, reserved for future image assets like:
  - Application logo
  - Icons
  - Sample images
  - Background images

## Usage

These static files are served by FastAPI through the `/static` route and are automatically mounted when the application starts.

Files can be accessed via:
- CSS: `/static/css/style.css`
- JS: `/static/js/app.js`
- Images: `/static/images/[filename]`
