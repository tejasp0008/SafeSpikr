#!/usr/bin/env python3
"""
Test Complete Sleep Detection System
Test the full system with AWS Rekognition
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sleep_detection_system import SleepDetectionSystem

def main():
    """Test the complete system"""
    print("🎯 Complete Sleep Detection System Test")
    print("=" * 40)
    print("This will test the full system with AWS Rekognition")
    print("\nInstructions:")
    print("1. Keep your eyes open for a few seconds")
    print("2. Close your eyes for 3+ seconds")
    print("3. Open your eyes again")
    print("4. Press Ctrl+C to stop")
    
    input("\nPress Enter to start...")
    
    # Initialize system
    system = SleepDetectionSystem()
    
    try:
        # Start monitoring
        if system.start_monitoring():
            print("\n✅ System started successfully!")
            print("🎬 Monitoring active - follow the instructions above")
            
            # Monitor for 60 seconds or until interrupted
            start_time = time.time()
            
            while time.time() - start_time < 60:
                # Get current state
                current_state = system.get_current_state()
                
                # Display status
                elapsed = time.time() - start_time
                print(f"\rTime: {elapsed:5.1f}s | State: {current_state.current_state.upper():10} | "
                      f"Confidence: {current_state.confidence:5.1f}% | "
                      f"Method: {current_state.detection_method:12} | "
                      f"Duration: {current_state.duration_in_state:4.1f}s", end="")
                
                time.sleep(0.5)
            
            print(f"\n\n⏰ Test completed after 60 seconds")
            
        else:
            print("❌ Failed to start system")
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Test stopped by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
    finally:
        # Get final statistics
        try:
            status = system.get_system_status()
            summary = system.get_detection_summary()
            
            print("\n" + "=" * 50)
            print("📊 TEST SUMMARY")
            print("=" * 50)
            print(f"Detection Method: {status['system']['detection_method']}")
            print(f"AWS Available: {status['services']['aws_available']}")
            print(f"Camera Available: {status['services']['camera_available']}")
            print(f"Total Detections: {summary['total_detections']}")
            print(f"Detection Accuracy: {status['performance']['detection_accuracy']:.1f}%")
            print(f"Processing FPS: {status['performance']['frames_per_second']:.1f}")
            
            if 'state_distribution' in summary:
                print(f"\nState Distribution:")
                for state, count in summary['state_distribution'].items():
                    print(f"  {state}: {count} detections")
            
        except Exception as e:
            print(f"Error getting summary: {e}")
        
        # Shutdown system
        system.shutdown()
        print("\n👋 System shutdown complete")

if __name__ == '__main__':
    main()