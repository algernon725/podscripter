#!/usr/bin/env python3
"""
Regression test for subdomain sentence splitting bug.

Ensures that domains with subdomain prefixes (www., blog., ftp., etc.) 
are not incorrectly split during Spanish transcription processing.

Bug was: "Visit www.google.com" became "Visit www. Google.com" and split into two sentences.
Fix: Enhanced domain masking to handle subdomain patterns before space insertion.
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation
from domain_utils import mask_domains, unmask_domains

def test_subdomain_sentence_splitting():
    """Test that subdomain patterns are not split during sentence processing."""
    
    test_cases = [
        # Original issue patterns
        ("Visit www.google.com for search", "Visit www.google.com for search."),
        ("Go to blog.example.org please", "Go to blog.example.org please."),
        ("Check ftp.downloads.net today", "Check ftp.downloads.net today."),
        ("Ve a espanolistos.com ahora", "Ve a espanolistos.com ahora."),
        ("Visita www.espanolistos.com por favor", "Visita www.espanolistos.com por favor."),
        
        # Additional subdomain patterns
        ("Visit mail.company.co.uk", "Visit mail.company.co.uk."),
        ("Check api.service.com", "Check api.service.com."),
        ("Go to cdn.assets.net", "Go to cdn.assets.net."),
        ("Visit help.support.org", "Visit help.support.org."),
        
        # Mixed content
        ("Visit www.google.com and then espanolistos.com", "Visit www.google.com and then espanolistos.com."),
    ]
    
    print("Testing subdomain sentence splitting...")
    
    failed_tests = []
    
    for original, expected in test_cases:
        result = restore_punctuation(original, language='es')
        
        # Check for splitting issues
        has_split_domain = any(pattern in result for pattern in [
            ". com", ". org", ". net", ". co.uk", ". com.br",
            "www. ", "blog. ", "ftp. ", "api. ", "cdn. "
        ])
        
        if has_split_domain:
            failed_tests.append((original, expected, result))
            print(f"❌ FAILED: {original}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")
        else:
            print(f"✅ PASSED: {original}")
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} test(s) failed!")
        return False
    else:
        print(f"\n✅ All {len(test_cases)} tests passed!")
        return True

def test_subdomain_masking():
    """Test that subdomain masking works correctly."""
    
    test_cases = [
        "Visit www.google.com",
        "Check blog.example.org", 
        "Go to ftp.downloads.net",
        "Visit mail.company.co.uk",
        "Check sinónimosonline.com"  # accented domain
    ]
    
    print("\nTesting subdomain masking...")
    
    for test_case in test_cases:
        masked = mask_domains(test_case, use_exclusions=True, language='es')
        unmasked = unmask_domains(masked)
        
        # Check if masking/unmasking preserved the original
        preserved = test_case == unmasked
        
        # Check if subdomain patterns are being masked
        has_subdomain = any(prefix in test_case for prefix in ['www.', 'blog.', 'ftp.', 'mail.'])
        subdomain_masked = '__DOT__' in masked and any(prefix in masked for prefix in ['www', 'blog', 'ftp', 'mail'])
        
        if not preserved:
            print(f"❌ MASKING FAILED: {test_case}")
            print(f"   Original: {test_case}")
            print(f"   Masked:   {masked}")
            print(f"   Unmasked: {unmasked}")
            return False
        elif has_subdomain and not subdomain_masked:
            print(f"❌ SUBDOMAIN NOT MASKED: {test_case}")
            return False
        else:
            print(f"✅ MASKING OK: {test_case}")
    
    return True

if __name__ == "__main__":
    success1 = test_subdomain_sentence_splitting()
    success2 = test_subdomain_masking()
    
    if success1 and success2:
        print("\n🎉 All subdomain tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
