package com.example.drawing_app

import android.app.Dialog
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.core.view.get

class MainActivity : AppCompatActivity() {

    private var drawingView : DrawingView? = null
    private var mImageButtonCurrentPaint : ImageButton? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        drawingView = findViewById(R.id.drawing_view)
        drawingView?.setBrushSize(20F)

        val llPaintColors = findViewById<LinearLayout>(R.id.ll_color_pallet)
        mImageButtonCurrentPaint = llPaintColors[1] as ImageButton

        mImageButtonCurrentPaint?.setImageDrawable(
            ContextCompat.getDrawable(this, R.drawable.pallet_selected)
        )



        val brushSizeSelectorBtn : ImageButton = findViewById(R.id.brush_size_selector)
        brushSizeSelectorBtn.setOnClickListener {
            showBrushSizeSelectorDialog()
        }
    }


    private fun showBrushSizeSelectorDialog(){
        val brushDialog = Dialog(this)

        brushDialog.setContentView(R.layout.brush_size_dialog)
        brushDialog.setTitle("Brush Size : ")

        val smallBrushBtn : ImageButton = brushDialog.findViewById(R.id.small_brush)
        val mediumBrushBtn : ImageButton = brushDialog.findViewById(R.id.medium_brush)
        val largeBrushBtn : ImageButton = brushDialog.findViewById(R.id.large_brush)

        smallBrushBtn.setOnClickListener {

            drawingView?.setBrushSize(10F)
            brushDialog.dismiss()
        }

        mediumBrushBtn.setOnClickListener {

            drawingView?.setBrushSize(20F)
            brushDialog.dismiss()

        }

        largeBrushBtn.setOnClickListener {

            drawingView?.setBrushSize(30F)
            brushDialog.dismiss()

        }

        brushDialog.show()

    }

    fun paintClicked(view : View){
//        Toast.makeText(this, "clicked paint",Toast.LENGTH_SHORT).show()
        if(view!==mImageButtonCurrentPaint){
            val imageButton = view as ImageButton
            val colorTag = imageButton.tag.toString()

            drawingView?.setColor(colorTag)


            imageButton.setImageDrawable(
                ContextCompat.getDrawable(this,R.drawable.pallet_selected)
            )

            mImageButtonCurrentPaint?.setImageDrawable(
            ContextCompat.getDrawable(this,R.drawable.pallet_normal)
            )
            mImageButtonCurrentPaint = view

        }
    }


}