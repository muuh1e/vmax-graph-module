#!/bin/bash
# Quick test script for GNN models

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== GNN Model Testing Script ===${NC}"
echo ""

# Parse arguments
MODE=${1:-quick}  # quick, full, or compare

case $MODE in
    quick)
        echo -e "${GREEN}Running quick test (50 records, 10 epochs)${NC}"
        echo "This will take about 5-10 minutes..."
        echo ""

        for model in simple_gnn typed_gnn hierarchical_gnn; do
            echo -e "${BLUE}Testing $model...${NC}"
            python gnn_pipeline/train/run.py \
                --model $model \
                --max_records 50 \
                --epochs 10 \
                --batch_size 8 \
                --checkpoint_dir checkpoints/quick_test/$model \
                --results_dir results/quick_test/$model
            echo ""
        done

        echo -e "${GREEN}Quick test complete!${NC}"
        echo "Results saved to: results/quick_test/"
        echo "Checkpoints saved to: checkpoints/quick_test/"
        ;;

    full)
        echo -e "${GREEN}Running full training (50 epochs)${NC}"
        echo "This will take 30-60 minutes per model..."
        echo ""

        MODEL=${2:-typed_gnn}
        PRESET=${3:-with_l2l}

        echo -e "${BLUE}Training $MODEL with preset $PRESET${NC}"
        python gnn_pipeline/train/run.py \
            --model $MODEL \
            --graph_preset $PRESET \
            --epochs 50 \
            --batch_size 16 \
            --hidden 256 \
            --checkpoint_dir checkpoints/$MODEL \
            --results_dir results/$MODEL

        echo -e "${GREEN}Training complete!${NC}"
        echo "Best model: checkpoints/$MODEL/best_model.pt"
        ;;

    compare)
        echo -e "${GREEN}Running comparison (all models, 30 epochs)${NC}"
        echo "This will take about 1-2 hours..."
        echo ""

        # Simple GNN baseline
        echo -e "${BLUE}1/3: Training SimpleHeteroGNN (baseline)${NC}"
        python gnn_pipeline/train/run.py \
            --model simple_gnn \
            --graph_preset baseline \
            --epochs 30 \
            --batch_size 16 \
            --checkpoint_dir checkpoints/compare/simple_gnn \
            --results_dir results/compare/simple_gnn

        # TypedHeteroGNN with L2L
        echo -e "${BLUE}2/3: Training TypedHeteroGNN (with L2L edges)${NC}"
        python gnn_pipeline/train/run.py \
            --model typed_gnn \
            --graph_preset with_l2l \
            --epochs 30 \
            --batch_size 16 \
            --checkpoint_dir checkpoints/compare/typed_gnn \
            --results_dir results/compare/typed_gnn

        # HierarchicalGNN with smart A2A
        echo -e "${BLUE}3/3: Training HierarchicalGNN (with smart A2A)${NC}"
        python gnn_pipeline/train/run.py \
            --model hierarchical_gnn \
            --graph_preset smart_a2a_k5 \
            --epochs 30 \
            --batch_size 16 \
            --hidden 256 \
            --checkpoint_dir checkpoints/compare/hierarchical_gnn \
            --results_dir results/compare/hierarchical_gnn

        echo ""
        echo -e "${GREEN}Comparison complete!${NC}"
        echo "Results saved to: results/compare/"
        echo ""
        echo "To compare results, check the final ADE/FDE values for each model."
        ;;

    *)
        echo "Usage: $0 [mode] [options]"
        echo ""
        echo "Modes:"
        echo "  quick              - Quick test (50 records, 10 epochs) for all models"
        echo "  full [model] [preset] - Full training (50 epochs) for one model"
        echo "  compare            - Train all models with same settings for comparison"
        echo ""
        echo "Examples:"
        echo "  $0 quick"
        echo "  $0 full typed_gnn with_l2l"
        echo "  $0 full hierarchical_gnn smart_a2a_k5"
        echo "  $0 compare"
        ;;
esac
