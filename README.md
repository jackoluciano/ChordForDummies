Berikut versi yang lebih sesuai sama gaya lu: direct, teknis secukupnya, tanpa basa-basi.

---

# Chord for Dummies

Proyek computer vision untuk menerjemahkan gesture tangan menjadi chord secara real-time menggunakan MediaPipe.

Dibuat karena beberapa musisi (terutama di konteks gereja) sering ngodein chord pakai tangan, tapi tidak semua orang paham sistemnya. Proyek ini mengubah gesture tersebut menjadi output chord yang eksplisit.

---

## Overview

Sistem ini tidak langsung memetakan gesture ke chord absolut, tapi menggunakan pendekatan berbasis interval.

Gesture tangan merepresentasikan derajat (I–VII), lalu sistem menghitung chord berdasarkan nada dasar (root) yang aktif. Nada dasar bisa berubah secara dinamis lewat gesture tertentu.

Dengan pendekatan ini, sistem tidak terikat pada satu key dan lebih fleksibel mengikuti konteks musik.

---

## Cara Kerja

1. Hand tracking dilakukan menggunakan MediaPipe untuk mendapatkan landmark jari secara real-time.

2. Gesture detection:

   * Kombinasi jari menentukan interval:

     * Index → I
     * Index + Middle → II
     * Index + Middle + Ring → III
     * Index + Middle + Ring + Pinky → IV
     * All fingers → V
     * Thumb → VI
     * Thumb + Index → VII

3. Modifier:

   * Gesture menghadap atas → natural
   * Gesture menghadap bawah → sharp (#)
   * III# dan VII# dianggap tidak valid

4. Transpose (perubahan root):

   * Middle finger (gerakan naik / “jump”) → transpose +1 per peak
   * Pinky (jump) → transpose -1 per peak
   * Peak dihitung dari perubahan arah gerakan (tanpa cooldown)

5. Perhitungan chord:

   * Menggunakan array chromatic:
     `['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']`
   * Interval dikonversi ke semitone offset
   * Chord dihitung dengan modular arithmetic terhadap root

---

## Limitations (V1)

* Deteksi gesture masih bisa salah, terutama saat posisi tangan tidak stabil
* Peak detection masih sensitif terhadap noise gerakan kecil
* Kadang output chord tidak sesuai akibat misclassification gesture

---

## Future Improvements

* Menambahkan ekspresi wajah untuk menentukan kualitas chord (misal major / minor)
* Tuning ulang gesture detection agar lebih stabil dan minim error

---
