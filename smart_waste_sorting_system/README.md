# Smart Waste Sorting System

A real-time waste detection and sorting system with industry mapping capabilities, integrated with Linear MCP for workflow management.

## Features

- **Real-time Camera Feed**: Access webcam for live waste detection
- **AI-Powered Detection**: Identify different types of waste (organic/inorganic)
- **Industry Mapping**: Get detailed information about recycling applications
- **Linear MCP Integration**: Track detections and create workflow issues
- **Modern Web Interface**: Beautiful, responsive design with real-time updates
- **Statistics Dashboard**: Track detection counts and recycling rates

## Supported Waste Types

### Organic Waste
- Banana Peel
- Orange Peel

### Inorganic Waste
- Plastic
- Paper
- Wood

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart_waste_sorting_system
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Linear MCP** (if not already configured)
   - Ensure your `mcp.json` file includes the Linear MCP server configuration
   - The system will work with mock data if Linear MCP is not available

## Usage

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your web browser**
   - Navigate to `http://localhost:5000`
   - Allow camera access when prompted

3. **Use the system**
   - Click "Start Camera" to begin video feed
   - Click "Capture & Analyze" to detect waste in the current frame
   - View detection results and industry applications
   - Monitor statistics in the dashboard

## API Endpoints

- `GET /` - Main web interface
- `POST /api/detect-waste` - Analyze waste from image data
- `GET /api/industry-info/<waste_type>` - Get industry information
- `GET /api/stats` - Get system statistics
- `GET /api/health` - Health check

## Linear MCP Integration

The system integrates with Linear MCP to:
- Create issues for waste detection events
- Track workflow progress
- Manage industry application data
- Generate reports and analytics

## Configuration

Edit `src/config.py` to customize:
- Camera settings (resolution, FPS)
- Model parameters
- API settings
- File paths

## Development

### Project Structure
```
smart_waste_sorting_system/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── web/                   # Frontend files
│   ├── index.html        # Main webpage
│   ├── styles.css        # CSS styling
│   └── script.js         # JavaScript functionality
├── src/                   # Source code
│   ├── config.py         # Configuration
│   └── linear_mcp_service.py  # Linear MCP integration
└── README.md             # This file
```

### Adding New Waste Types

1. Update `CLASSES` in `src/config.py`
2. Add industry mapping in `src/linear_mcp_service.py`
3. Update the detection model (if using ML)

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

**Note**: Camera access requires HTTPS in production environments.

## Troubleshooting

### Camera Access Issues
- Ensure browser has camera permissions
- Check if camera is being used by another application
- Try refreshing the page

### Detection Errors
- Check browser console for JavaScript errors
- Verify Flask server is running
- Check network connectivity

### Linear MCP Issues
- Verify MCP configuration in `mcp.json`
- Check Linear API credentials
- Review server logs for MCP connection errors

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub