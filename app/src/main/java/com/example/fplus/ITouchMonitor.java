/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Using: C:\\Users\\57615\\AppData\\Local\\Android\\Sdk\\build-tools\\37.0.0\\aidl.exe -I c:\\Users\\57615\\AndroidStudioProjects\\FPlus\\app\\src\\main\\aidl -o c:\\Users\\57615\\AndroidStudioProjects\\FPlus\\poc\\aidl_out3 c:\\Users\\57615\\AndroidStudioProjects\\FPlus\\app\\src\\main\\aidl\\com\\example\\fplus\\ITouchMonitor.aidl
 *
 * DO NOT CHECK THIS FILE INTO A CODE TREE (e.g. git, etc..).
 * ALWAYS GENERATE THIS FILE FROM UPDATED AIDL COMPILER
 * AS A BUILD INTERMEDIATE ONLY. THIS IS NOT SOURCE CODE.
 */
package com.example.fplus;
/**
 * 触摸屏监听服务接口（运行在 Shizuku UserService 进程，shell UID）。
 * 只读打开真实触摸屏，用于判断用户是否正用手触摸屏幕，供"检测让路"使用。
 * 
 * 注意：destroy() 的 transaction code 必须为 16777114（Shizuku 约定），
 * Shizuku 服务停止时会用固定 transaction code 调用它做清理。
 */
public interface ITouchMonitor extends android.os.IInterface
{
  /** Default implementation for ITouchMonitor. */
  public static class Default implements com.example.fplus.ITouchMonitor
  {
    /** Shizuku 保留的销毁方法，transaction code 固定 16777114 */
    @Override public void destroy() throws android.os.RemoteException
    {
    }
    /** 开始监听真实触摸屏 */
    @Override public void start() throws android.os.RemoteException
    {
    }
    /** 用户是否正用手触摸屏幕 */
    @Override public boolean isUserTouching() throws android.os.RemoteException
    {
      return false;
    }
    /** 手指当前归一化 X 坐标（0~1，相对触摸屏原始坐标系） */
    @Override public float getTouchX() throws android.os.RemoteException
    {
      return 0.0f;
    }
    /** 手指当前归一化 Y 坐标（0~1，相对触摸屏原始坐标系） */
    @Override public float getTouchY() throws android.os.RemoteException
    {
      return 0.0f;
    }
    @Override
    public android.os.IBinder asBinder() {
      return null;
    }
  }
  /** Local-side IPC implementation stub class. */
  public static abstract class Stub extends android.os.Binder implements com.example.fplus.ITouchMonitor
  {
    /** Construct the stub and attach it to the interface. */
    @SuppressWarnings("this-escape")
    public Stub()
    {
      this.attachInterface(this, DESCRIPTOR);
    }
    /**
     * Cast an IBinder object into an com.example.fplus.ITouchMonitor interface,
     * generating a proxy if needed.
     */
    public static com.example.fplus.ITouchMonitor asInterface(android.os.IBinder obj)
    {
      if ((obj==null)) {
        return null;
      }
      android.os.IInterface iin = obj.queryLocalInterface(DESCRIPTOR);
      if (((iin!=null)&&(iin instanceof com.example.fplus.ITouchMonitor))) {
        return ((com.example.fplus.ITouchMonitor)iin);
      }
      return new com.example.fplus.ITouchMonitor.Stub.Proxy(obj);
    }
    @Override public android.os.IBinder asBinder()
    {
      return this;
    }
    @Override public boolean onTransact(int code, android.os.Parcel data, android.os.Parcel reply, int flags) throws android.os.RemoteException
    {
      if (code >= android.os.IBinder.FIRST_CALL_TRANSACTION && code <= android.os.IBinder.LAST_CALL_TRANSACTION) {
        data.enforceInterface(DESCRIPTOR);
      }
      switch (code)
      {
        case TRANSACTION_destroy:
        {
          this.destroy();
          reply.writeNoException();
          break;
        }
        case TRANSACTION_start:
        {
          this.start();
          reply.writeNoException();
          break;
        }
        case TRANSACTION_isUserTouching:
        {
          boolean _result = this.isUserTouching();
          reply.writeNoException();
          reply.writeInt(((_result)?(1):(0)));
          break;
        }
        case TRANSACTION_getTouchX:
        {
          float _result = this.getTouchX();
          reply.writeNoException();
          reply.writeFloat(_result);
          break;
        }
        case TRANSACTION_getTouchY:
        {
          float _result = this.getTouchY();
          reply.writeNoException();
          reply.writeFloat(_result);
          break;
        }
        default:
        {
          return super.onTransact(code, data, reply, flags);
        }
      }
      return true;
    }
    private static final class Proxy implements com.example.fplus.ITouchMonitor
    {
      private android.os.IBinder mRemote;
      Proxy(android.os.IBinder remote)
      {
        mRemote = remote;
      }
      @Override public android.os.IBinder asBinder()
      {
        return mRemote;
      }
      public final java.lang.String getInterfaceDescriptor()
      {
        return DESCRIPTOR;
      }
      /** Shizuku 保留的销毁方法，transaction code 固定 16777114 */
      @Override public void destroy() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_destroy, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      /** 开始监听真实触摸屏 */
      @Override public void start() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_start, _data, _reply, 0);
          _reply.readException();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
      }
      /** 用户是否正用手触摸屏幕 */
      @Override public boolean isUserTouching() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        boolean _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_isUserTouching, _data, _reply, 0);
          _reply.readException();
          _result = (0!=_reply.readInt());
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
      /** 手指当前归一化 X 坐标（0~1，相对触摸屏原始坐标系） */
      @Override public float getTouchX() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        float _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_getTouchX, _data, _reply, 0);
          _reply.readException();
          _result = _reply.readFloat();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
      /** 手指当前归一化 Y 坐标（0~1，相对触摸屏原始坐标系） */
      @Override public float getTouchY() throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        android.os.Parcel _reply = android.os.Parcel.obtain();
        float _result;
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          boolean _status = mRemote.transact(Stub.TRANSACTION_getTouchY, _data, _reply, 0);
          _reply.readException();
          _result = _reply.readFloat();
        }
        finally {
          _reply.recycle();
          _data.recycle();
        }
        return _result;
      }
    }
    static final int TRANSACTION_destroy = (android.os.IBinder.FIRST_CALL_TRANSACTION + 16777114);
    static final int TRANSACTION_start = (android.os.IBinder.FIRST_CALL_TRANSACTION + 1);
    static final int TRANSACTION_isUserTouching = (android.os.IBinder.FIRST_CALL_TRANSACTION + 2);
    static final int TRANSACTION_getTouchX = (android.os.IBinder.FIRST_CALL_TRANSACTION + 3);
    static final int TRANSACTION_getTouchY = (android.os.IBinder.FIRST_CALL_TRANSACTION + 4);
  }
  /** @hide */
  public static final java.lang.String DESCRIPTOR = "com.example.fplus.ITouchMonitor";
  /** Shizuku 保留的销毁方法，transaction code 固定 16777114 */
  public void destroy() throws android.os.RemoteException;
  /** 开始监听真实触摸屏 */
  public void start() throws android.os.RemoteException;
  /** 用户是否正用手触摸屏幕 */
  public boolean isUserTouching() throws android.os.RemoteException;
  /** 手指当前归一化 X 坐标（0~1，相对触摸屏原始坐标系） */
  public float getTouchX() throws android.os.RemoteException;
  /** 手指当前归一化 Y 坐标（0~1，相对触摸屏原始坐标系） */
  public float getTouchY() throws android.os.RemoteException;
}
