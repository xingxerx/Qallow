#!/bin/bash

# IBM Quantum Credentials Setup Script

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              🔑 IBM QUANTUM CREDENTIALS SETUP 🔑                           ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 INSTRUCTIONS:"
echo "1. Go to https://quantum.ibm.com/"
echo "2. Sign up (free account)"
echo "3. Go to Account settings"
echo "4. Copy your API key"
echo "5. Copy your CRN (Cloud Resource Name)"
echo ""

read -p "Enter your IBM Quantum API key: " API_KEY
read -p "Enter your IBM Quantum CRN: " CRN

if [ -z "$API_KEY" ] || [ -z "$CRN" ]; then
    echo "❌ Error: API key and CRN are required"
    exit 1
fi

echo ""
echo "Setting environment variables..."

export QISKIT_IBM_TOKEN="$API_KEY"
export QISKIT_IBM_INSTANCE="$CRN"

echo "export QISKIT_IBM_TOKEN=\"$API_KEY\"" >> ~/.bashrc
echo "export QISKIT_IBM_INSTANCE=\"$CRN\"" >> ~/.bashrc

echo ""
echo "✅ Credentials saved to ~/.bashrc"
echo ""
echo "To use immediately, run:"
echo "  source ~/.bashrc"
echo ""
echo "To verify credentials are set:"
echo "  echo \$QISKIT_IBM_TOKEN"
echo "  echo \$QISKIT_IBM_INSTANCE"
echo ""

