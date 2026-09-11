# 🌾 Krishi Yukti

**AI-Powered Agriculture Ecosystem**

Krishi Yukti is an end-to-end smart agriculture platform designed to help farmers with crop health, crop recommendations, yield intelligence, weather insights, agricultural products, and farmer-to-buyer commerce.

## Vision

> One platform for a farmer to understand the farm, protect the crop, buy what is needed, and sell the harvest.

## Core Modules

- 🌱 **AI Crop Doctor** — image-based crop disease detection and health guidance
- 🌾 **Crop Recommendation** — recommend suitable crops from soil and environmental conditions
- 📈 **Yield Prediction** — estimate expected crop yield
- ☁️ **Weather & Climate Intelligence** — weather-aware farm risk insights
- 🛒 **Agri Store** — seeds, fertilizers, equipment and other farm supplies
- 🤝 **Farmer Marketplace** — farmer listings, buyers, offers and crop sales
- 💬 **Farmer–Buyer Communication** — messaging and negotiation
- 📊 **Farmer Dashboard** — unified farm and AI insights
- 🔐 **Role-based Authentication** — farmer, buyer and administrator roles

## Planned Architecture

```text
React + Bootstrap Frontend
          │
          ▼
      FastAPI Backend
          │
    ┌─────┼──────────────┐
    ▼     ▼              ▼
  Core   AI Services   Weather
  API                 Integration
    │     │
    │   ┌─┼─────────────┐
    │   ▼ ▼             ▼
    │ Disease       Crop/Yield
    │ Detection      Models
    │
    ▼
 PostgreSQL
    │
 Redis / Object Storage
```

## Repository Structure

```text
krishi-yukti/
├── backend/                 # FastAPI application
├── frontend/                # React + Bootstrap application
├── ml/                      # Training and inference code
│   ├── disease/             # Crop disease computer vision
│   ├── crop_recommendation/ # Crop recommendation models
│   └── yield_prediction/    # Yield prediction models
├── database/                # Database schema and seed data
├── docs/                    # Architecture and project documentation
├── tests/                   # Backend and ML tests
├── docker/                  # Container and deployment configuration
├── .github/workflows/       # CI/CD workflows
└── README.md
```

## Development Principles

1. Build a working MVP first.
2. Keep AI predictions explainable and expose confidence/uncertainty.
3. Keep a human-in-the-loop for agricultural and health-related recommendations.
4. Never present AI output as a guaranteed diagnosis or blindly prescribe regulated chemicals.
5. Separate ML services from transactional application APIs so the system can scale independently.
6. Add automated tests and CI from the beginning.

## Initial Milestones

- [x] Repository initialization
- [ ] Backend application skeleton
- [ ] React application skeleton
- [ ] PostgreSQL data model
- [ ] Authentication and roles
- [ ] Crop disease ML pipeline
- [ ] Crop recommendation model
- [ ] Yield prediction model
- [ ] Weather integration
- [ ] Agri marketplace
- [ ] Farmer-to-buyer marketplace
- [ ] Messaging
- [ ] Dockerized deployment
- [ ] CI/CD
- [ ] Prometheus + Grafana monitoring

## Disclaimer

Krishi Yukti is intended as a decision-support and educational platform. AI predictions should be reviewed by qualified agricultural professionals where appropriate. The platform should not be treated as a substitute for professional medical, agricultural, or regulatory advice.
