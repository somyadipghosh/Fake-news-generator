"""
Data Augmentation Script - Fetch real news and create balanced dataset
"""
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import RAW_DATA_DIR
from src.news_fetcher import NewsAPIFetcher


# Fake news examples with patterns typical of misinformation
FAKE_NEWS_EXAMPLES = [
    {
        'headline': 'SHOCKING! Scientists Discover Aliens Living Underground! Government Cover-Up EXPOSED!',
        'text': 'BREAKING NEWS!!! Scientists have FINALLY revealed what the government has been hiding for DECADES! Aliens are living underground and they DON\'T want you to know! This SHOCKING discovery will change EVERYTHING we thought we knew! Experts are STUNNED by these findings! The mainstream media is trying to SILENCE this story! You won\'t BELIEVE what they found! Share this before it gets DELETED! This is the TRUTH they don\'t want you to see! Act NOW before it\'s too late! EVERYONE needs to know about this IMMEDIATELY!',
        'label': 1
    },
    {
        'headline': 'Miracle Diet Pill Melts 50 Pounds in 1 Week! Doctors Hate This Simple Trick!',
        'text': 'AMAZING breakthrough in weight loss! This ONE weird trick will INSTANTLY melt away fat! Lose 50 pounds in just ONE WEEK without diet or exercise! Big Pharma is FURIOUS and trying to BAN this miracle pill! Thousands of people are using this SECRET method! You won\'t BELIEVE the results! This REVOLUTIONARY discovery changes EVERYTHING! Click NOW to get your FREE sample before they\'re gone! Limited time offer! HURRY! This deal won\'t last long!',
        'label': 1
    },
    {
        'headline': 'URGENT! New Technology Allows Government to Read Your Thoughts Through WiFi!',
        'text': 'BREAKING: Security experts have EXPOSED a terrifying new technology! Your WiFi router is being used to READ YOUR THOUGHTS! This is NOT a joke! The government has been secretly monitoring your brain waves through your internet connection! Whistleblowers have REVEALED this shocking truth! Protect yourself NOW before it\'s too late! Share this with EVERYONE you know! This information is being CENSORED! They don\'t want you to know! Wake up people! The truth is out there!',
        'label': 1
    },
    {
        'headline': 'Celebrity DEAD at 25! Shocking Details REVEALED! You Won\'t Believe What Happened!',
        'text': 'TRAGIC news breaking RIGHT NOW! Famous celebrity found DEAD in mysterious circumstances! Police are COVERING UP the real cause! Insider sources reveal SHOCKING details that will BLOW YOUR MIND! This story is developing and getting CRAZIER by the minute! What they\'re not telling you will SHOCK you! The TRUTH is finally coming out! Share this before mainstream media DELETES it! This is HUGE! EVERYONE is talking about this!',
        'label': 1
    },
    {
        'headline': 'Ancient Prophecy Predicts World Will End Next Month! Prepare NOW!',
        'text': 'APOCALYPTIC prediction from ancient texts CONFIRMS the end is near! Scholars have DECODED mysterious writings that predict CATASTROPHIC events next month! This 5000-year-old prophecy is COMING TRUE before our eyes! Signs are EVERYWHERE! Wake up sheeple! The elite are already preparing their bunkers! You NEED to see this! Government is HIDING the truth! Stock up on supplies NOW! This is your FINAL WARNING! Don\'t say we didn\'t warn you!',
        'label': 1
    },
    {
        'headline': 'EXPOSED: Tap Water Contains Mind Control Chemicals! The Truth They Hide!',
        'text': 'SHOCKING investigation reveals what\'s REALLY in your tap water! Government secretly adding MIND CONTROL chemicals to the water supply! This is NOT a conspiracy theory - we have PROOF! Whistleblowers risk their lives to bring you this information! Big corporations are POISONING us! The fluoride lie EXPOSED! Protect your family NOW! Install water filters TODAY! They want to keep us DOCILE and OBEDIENT! Wake up! Time is running out!',
        'label': 1
    },
    {
        'headline': '5G Towers Causing Cancer! Millions at Risk! Scientists Confirm Deadly Effects!',
        'text': 'URGENT WARNING! New study PROVES 5G technology is KILLING people! Cancer rates SKYROCKET near 5G towers! Telecom companies are COVERING UP the evidence! Your children are in DANGER! Scientists who spoke out were SILENCED! This is the biggest health crisis of our time! Share this vital information before it\'s DELETED! Big Tech doesn\'t want you to know! Radiation levels are OFF THE CHARTS! Protect yourself and your loved ones NOW!',
        'label': 1
    },
    {
        'headline': 'Billionaire Reveals One Simple Investment Trick! Become Rich Overnight!',
        'text': 'Self-made billionaire EXPOSES the ONE investment secret Wall Street doesn\'t want you to know! Make MILLIONS with this simple trick! No experience needed! Start earning $10,000 per day from home! This REVOLUTIONARY system is making ordinary people RICH! Banks HATE this method! Limited spots available! Act FAST before this opportunity disappears! Join thousands of people already making fortunes! Click NOW to claim your spot! Don\'t miss out on this life-changing opportunity!',
        'label': 1
    },
    {
        'headline': 'Vaccines Contain Microchips! Ultimate Proof of Global Control Agenda!',
        'text': 'BOMBSHELL evidence CONFIRMS vaccines contain tracking microchips! This is the NEW WORLD ORDER in action! They want to track and control EVERYONE! Former insider EXPOSES the sinister plot! Your freedom is being STOLEN! The mainstream media is complicit in this MASSIVE cover-up! Independent researchers have found UNDENIABLE proof! Wake up before it\'s too late! Share this with EVERYONE! The truth movement is growing! Together we can STOP this tyranny!',
        'label': 1
    },
    {
        'headline': 'Free Energy Device Invented! Oil Companies Trying to Suppress Amazing Technology!',
        'text': 'REVOLUTIONARY invention provides UNLIMITED free energy! Genius inventor creates device that generates power from NOTHING! Oil companies are TERRIFIED and trying to BUY the patent to BURY it forever! This changes EVERYTHING! Energy bills could be ZERO! Big Oil is desperate to HIDE this from the public! The inventor\'s life is in DANGER! Download the blueprints NOW before they\'re DELETED! Fight back against corporate greed! Free energy for ALL!',
        'label': 1
    },
    {
        'headline': 'Moon Landing Was FAKE! New Evidence Proves Hollywood Hoax!',
        'text': 'STUNNING new analysis PROVES moon landing was filmed in Hollywood studio! NASA has been LYING to us for 50 years! Photo experts reveal UNDENIABLE evidence of fakery! The flag was waving - there\'s NO WIND on the moon! Shadows don\'t match up! Stanley Kubrick directed the whole thing! Whistleblowers coming forward with SHOCKING testimony! This is the BIGGEST lie in history! Government admits NOTHING! Wake up and see the TRUTH! Share this forbidden knowledge!',
        'label': 1
    },
    {
        'headline': 'Eating Chocolate Cures ALL Diseases! Doctors AMAZED by Miracle Food!',
        'text': 'INCREDIBLE discovery shows chocolate is the ULTIMATE superfood! Cures cancer, diabetes, heart disease, and MORE! Medical establishment is FURIOUS because this natural remedy makes their drugs OBSOLETE! Eat chocolate three times a day and NEVER get sick again! Pharmaceutical companies are trying to SUPPRESS this information! Indigenous tribes have known this SECRET for centuries! Scientists CONFIRM chocolate activates miraculous healing powers! Stock up NOW! Your health depends on it!',
        'label': 1
    },
    {
        'headline': 'Time Traveler From 2050 Warns Humanity! Apocalyptic Events Coming Soon!',
        'text': 'VERIFIED time traveler shares TERRIFYING warnings about our future! He passed LIE DETECTOR tests! Everything he predicted so far has come TRUE! Major disasters heading our way! The government is INTERROGATING him! Mainstream media is IGNORING this crucial story! His predictions are SPECIFIC and DETAILED! We have less than a year to prepare! This is NOT science fiction - this is REAL! Listen to his warnings before it\'s too late! The future is GRIM unless we ACT NOW!',
        'label': 1
    },
    {
        'headline': 'Bananas Going EXTINCT Next Week! Stock Up NOW Before They Disappear Forever!',
        'text': 'URGENT alert! Global banana shortage will hit NEXT WEEK! Scientists predict TOTAL extinction of bananas! Supermarkets will be EMPTY! Fungal disease spreading at unprecedented rate! Media BLACKOUT on this crisis! Food shortage panic buying already beginning! Get your bananas TODAY while you still can! Prices will SKYROCKET! This affects EVERYONE! Banana supply chains COLLAPSING worldwide! Expert warns this could trigger GLOBAL FAMINE! Don\'t be caught unprepared!',
        'label': 1
    },
    {
        'headline': 'Your Phone Camera is Spying on You 24/7! Security Expert Exposes Scandal!',
        'text': 'ALARMING revelation! Your smartphone camera is ALWAYS watching you - even when turned off! Security researchers discover SHOCKING backdoor! Every major phone manufacturer is COMPLICIT! Your private moments are being RECORDED and sent to secret servers! The surveillance state is HERE! Cover your camera IMMEDIATELY! This is WORSE than we imagined! Hackers and governments both exploiting this vulnerability! Protect your privacy NOW! This story is being CENSORED!',
        'label': 1
    }
]


def create_augmented_dataset(num_real_news=100, include_existing=True):
    """
    Create augmented dataset with real news from NewsAPI and fake news examples
    
    Args:
        num_real_news: Number of real news articles to fetch
        include_existing: Whether to include existing dataset
        
    Returns:
        DataFrame with combined data
    """
    print("\n" + "="*70)
    print("        DATA AUGMENTATION - BUILDING BETTER TRAINING SET")
    print("="*70)
    
    # Load existing data if requested
    all_data = []
    
    if include_existing:
        existing_path = Path(RAW_DATA_DIR) / 'sample_news.csv'
        if existing_path.exists():
            print(f"\n📂 Loading existing dataset...")
            existing_df = pd.read_csv(existing_path)
            print(f"   ✓ Loaded {len(existing_df)} existing articles")
            all_data.append(existing_df)
    
    # Fetch real news from NewsAPI
    print(f"\n🌐 Fetching {num_real_news} real news articles from NewsAPI...")
    try:
        fetcher = NewsAPIFetcher()
        real_news_df = fetcher.fetch_diverse_news(num_articles=num_real_news, delay=1)
        
        if len(real_news_df) > 0:
            print(f"   ✓ Successfully fetched {len(real_news_df)} real news articles")
            all_data.append(real_news_df)
        else:
            print("   ⚠️  No articles fetched from NewsAPI")
            
    except Exception as e:
        print(f"   ⚠️  Error fetching from NewsAPI: {e}")
        print("   Continuing with existing data and fake examples...")
    
    # Add fake news examples
    print(f"\n📝 Adding {len(FAKE_NEWS_EXAMPLES)} fake news examples...")
    fake_df = pd.DataFrame(FAKE_NEWS_EXAMPLES)
    all_data.append(fake_df)
    print(f"   ✓ Added fake news examples")
    
    # Combine all data
    print(f"\n⚙️  Combining datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Clean and standardize
    combined_df = combined_df[['headline', 'text', 'label']].copy()
    combined_df = combined_df.dropna()
    combined_df = combined_df.drop_duplicates(subset=['headline'], keep='first')
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total articles: {len(combined_df)}")
    print(f"   Real news (label=0): {len(combined_df[combined_df['label']==0])}")
    print(f"   Fake news (label=1): {len(combined_df[combined_df['label']==1])}")
    print(f"   Balance ratio: {len(combined_df[combined_df['label']==0]) / len(combined_df[combined_df['label']==1]):.2f}:1")
    
    # Save augmented dataset
    output_path = Path(RAW_DATA_DIR) / 'augmented_news.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved augmented dataset to: {output_path}")
    
    # Also update improved_news.csv
    improved_path = Path(RAW_DATA_DIR) / 'improved_news.csv'
    combined_df.to_csv(improved_path, index=False)
    print(f"💾 Updated: {improved_path}")
    
    return combined_df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment training dataset with real news from NewsAPI')
    parser.add_argument('--num-articles', type=int, default=100,
                       help='Number of real news articles to fetch (default: 100)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip existing dataset, start fresh')
    
    args = parser.parse_args()
    
    # Create augmented dataset
    df = create_augmented_dataset(
        num_real_news=args.num_articles,
        include_existing=not args.skip_existing
    )
    
    print("\n" + "="*70)
    print("                    AUGMENTATION COMPLETE!")
    print("="*70)
    print(f"\n✅ Dataset ready with {len(df)} articles")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the data: data/raw/augmented_news.csv")
    print(f"   2. Train the model: python quick_train.py")
    print(f"   3. Test predictions: python demo.py")
    print()


if __name__ == "__main__":
    main()
