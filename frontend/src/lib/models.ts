// Placeholder list: the backend still hardcodes its model
// (backend/agent/turn.py MODEL_ID) and the selection is not sent anywhere
// yet. Swap in real gateway models once POST /api/chat accepts one.
export const MODELS = [{ id: "placeholder", name: "This is a placeholder" }]

export const DEFAULT_MODEL = MODELS[0].id

export interface GatewayModel {
  id: string
  name: string
}
