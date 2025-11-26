"""Quick script to check Minari dataset size."""
import minari

try:
    dataset = minari.load_dataset('atari/pong/expert-v0')
    print(f"Dataset: atari/pong/expert-v0")
    print(f"Total episodes: {dataset.total_episodes}")
    print(f"Total steps: {dataset.total_steps}")
    print(f"\nChecking first few episodes...")
    
    episode_count = 0
    total_steps = 0
    for episode in dataset:
        episode_count += 1
        steps = len(episode.observations)
        total_steps += steps
        print(f"  Episode {episode.id}: {steps} steps")
        if episode_count >= 5:
            print(f"  ... (showing first 5)")
            break
    
    print(f"\nVerified: {episode_count} episodes checked")
    print(f"Total steps from checked episodes: {total_steps}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying to list available datasets...")
    try:
        available = minari.list_remote_datasets()
        pong_datasets = [d for d in available if 'pong' in d.lower()]
        print(f"\nPong-related datasets found: {len(pong_datasets)}")
        for ds in pong_datasets[:10]:
            print(f"  - {ds}")
    except Exception as e2:
        print(f"Could not list datasets: {e2}")

